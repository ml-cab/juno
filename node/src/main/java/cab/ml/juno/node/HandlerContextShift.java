/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.node;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KvContextShift;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * The parts of {@link ForwardPassHandler#shiftKv} every handler shares: shift the
 * request's host KV (the authoritative copy) with the handler's own rotation,
 * then rebuild any device mirror from it, so the mirror's written-prefix
 * watermark again covers exactly the shifted history.
 */
final class HandlerContextShift {

	private static final Logger log = Logger.getLogger(HandlerContextShift.class.getName());
	private static final AtomicBoolean mirrorWarned = new AtomicBoolean();

	private HandlerContextShift() {
	}

	/**
	 * Shifts the request's host KV in place.
	 *
	 * @throws IllegalStateException when the handler holds no KV for the request
	 */
	static void shiftHost(Map<String, SessionKvTensor[]> kvCacheK, Map<String, SessionKvTensor[]> kvCacheV,
			String requestId, int seqLen, int keep, int discard, RopeShift rope, int numKvHeads) {
		SessionKvTensor[] k = kvCacheK.get(requestId);
		SessionKvTensor[] v = kvCacheV.get(requestId);
		if (k == null || v == null)
			throw new IllegalStateException("context shift: no KV held for request " + requestId);
		KvContextShift.shift(k, v, seqLen, keep, discard, rope.back(discard, numKvHeads));
	}

	/**
	 * Rewrites every mirror that was in use (live, with rows written) from the
	 * shifted host KV of the same layer. A mirror the card cannot hold is retired;
	 * the request continues with host attention, which has the same history.
	 *
	 * @param mirrors one per layer, entries may be {@code null}; {@code null} for none
	 */
	static void rewriteMirrors(String handler, DeviceKvCache[] mirrors, SessionKvTensor[] k, SessionKvTensor[] v,
			int newLen) {
		if (mirrors == null)
			return;
		for (int li = 0; li < mirrors.length; li++) {
			DeviceKvCache m = mirrors[li];
			if (m == null || !m.live() || m.validTokens() == 0)
				continue;
			try {
				m.replacePrefix(k[li].toFloatArray(newLen), v[li].toFloatArray(newLen), newLen);
			} catch (IllegalStateException ex) {
				if (!GpuLayerOffload.isVramOom(ex))
					throw ex;
				m.close();
				if (mirrorWarned.compareAndSet(false, true))
					log.warning(handler + ": out of device memory rewriting the attention KV mirror after a context"
							+ " shift - attention continues on the CPU for this request, which holds the same history.");
			}
		}
	}
}
