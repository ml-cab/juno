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
import java.util.function.IntPredicate;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.SessionKvLayout;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * What every handler's {@link ForwardPassHandler#shareKvPrefix} does: give one request's KV map entry the first rows
 * of another's ({@link SessionKvLayout#sharePrefixK}), then copy those rows into the new request's device attention
 * KV, if it has one, so its attention stays on the device.
 */
final class HandlerKvShare {

	private static final Logger log = Logger.getLogger(HandlerKvShare.class.getName());
	private static final AtomicBoolean mirrorWarned = new AtomicBoolean();

	private HandlerKvShare() {
	}

	/**
	 * Whether a handler may share. A device attention KV is FP16 of the rows the request computed; the host rows
	 * equal those rows only when the host stores them unquantized, so with a quantized host KV a shared prefix's
	 * device rows would differ from the ones the request computes itself, and sharing is refused.
	 */
	static boolean supported(SessionKvLayout layout, boolean deviceAttention, NodeKVCacheAdapter adapter) {
		if (adapter != null)
			return false; // the adapter's copy would lack the shared rows
		if (!deviceAttention)
			return true;
		return layout.cacheTypes().typeK() == KvElementType.F16 && layout.cacheTypes().typeV() == KvElementType.F16;
	}

	/** Host bytes one position takes in {@code layers} layers of this layout, K and V. */
	static long bytesPerToken(SessionKvLayout layout, int layers) {
		return (long) layers * (bytes(layout.cacheTypes().typeK(), layout.kvDim())
				+ bytes(layout.cacheTypes().typeV(), layout.kvDim()));
	}

	private static long bytes(KvElementType type, int kvDim) {
		return switch (type) {
		case F16 -> (long) kvDim * Float.BYTES; // held as floats on the host
		case Q8_0 -> cab.ml.juno.kvcache.Q8_0KvCodec.encodedBytes(kvDim);
		};
	}

	/**
	 * Puts under {@code toId} new K and V layers whose positions {@code [0, len)} hold {@code fromId}'s rows,
	 * releasing what {@code toId} held. False, changing nothing, when {@code fromId} holds no KV or fewer than
	 * {@code len} rows.
	 */
	static boolean shareHost(Map<String, SessionKvTensor[]> kvCacheK, Map<String, SessionKvTensor[]> kvCacheV,
			SessionKvLayout layout, String fromId, String toId, int len) {
		if (fromId == null || toId == null || fromId.equals(toId) || len < 1)
			return false;
		SessionKvTensor[] k = kvCacheK.get(fromId);
		SessionKvTensor[] v = kvCacheV.get(fromId);
		if (k == null || v == null)
			return false;
		SessionKvTensor[] sharedK = layout.sharePrefixK(k, len);
		if (sharedK == null)
			return false;
		SessionKvTensor[] sharedV = layout.sharePrefixV(v, len);
		if (sharedV == null) {
			SessionKvLayout.releaseLayers(sharedK);
			return false;
		}
		SessionKvLayout.releaseLayers(kvCacheK.put(toId, sharedK));
		SessionKvLayout.releaseLayers(kvCacheV.put(toId, sharedV));
		return true;
	}

	/**
	 * Writes positions {@code [0, len)} of the request's host rows into its device mirrors on the layers that attend
	 * on the device, one transfer per tensor and layer; each mirror's watermark becomes {@code len}. A mirror that
	 * runs out of device memory is closed and that layer attends on the CPU, which holds the same rows.
	 */
	static void uploadMirrors(String handler, DeviceKvCache[] mirrors, SessionKvTensor[] k, SessionKvTensor[] v,
			int len, IntPredicate onDevice) {
		if (mirrors == null)
			return;
		for (int li = 0; li < mirrors.length; li++) {
			DeviceKvCache m = mirrors[li];
			if (m == null || !m.live() || !onDevice.test(li))
				continue;
			try {
				m.replacePrefix(k[li].toFloatArray(len), v[li].toFloatArray(len), len);
			} catch (IllegalStateException ex) {
				if (!GpuLayerOffload.isVramOom(ex))
					throw ex;
				m.close();
				if (mirrorWarned.compareAndSet(false, true))
					log.warning(handler + ": out of device memory copying a shared prompt prefix into the attention"
							+ " KV mirror - attention continues on the CPU for this request, which holds the same"
							+ " rows.");
			}
		}
	}
}
