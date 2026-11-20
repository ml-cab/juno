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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import cab.ml.juno.kvcache.KvSessionFile;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.kvcache.SessionKvLayout;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * What every host-KV handler's {@link ForwardPassHandler#saveKv} and {@link ForwardPassHandler#restoreKv} share: one
 * request's layers to a {@link KvSessionFile} and back into new layers of the handler's own layout. The handler then
 * puts the restored layers under the request id and rebuilds its device attention rows from them, as
 * {@link ForwardPassHandler#shareKvPrefix} does.
 */
final class HandlerKvSnapshot {

	private HandlerKvSnapshot() {
	}

	/** Restored layers and the file's size. */
	record Restored(SessionKvTensor[] k, SessionKvTensor[] v, long bytes) {
	}

	/** Writes {@code requestId}'s rows {@code [0, stamp.positions())}; refused when the handler holds none for it. */
	static long save(String handler, Map<String, SessionKvTensor[]> kvCacheK, Map<String, SessionKvTensor[]> kvCacheV,
			String requestId, KvSessionStamp stamp, Path file) throws IOException {
		SessionKvTensor[] k = kvCacheK.get(requestId);
		SessionKvTensor[] v = kvCacheV.get(requestId);
		if (k == null || v == null)
			throw new KvSessionFileException(Reason.REFUSED, handler + " holds no KV for " + requestId);
		return KvSessionFile.write(file, stamp, k, v);
	}

	/**
	 * Refuses a restore whose device attention rows, rebuilt from the file's host rows, would not equal the rows the
	 * saved session wrote itself: the device KV keeps FP16 beside a quantized host KV
	 * ({@link DeviceKvFormats#holds}), the refusal {@link HandlerKvShare#supported} makes for a shared prefix.
	 *
	 * @param deviceFormats the handler's device KV formats, or {@code null} without GPU attention
	 */
	static void requireReproducible(String handler, SessionKvLayout layout, DeviceKvFormats deviceFormats) {
		if (deviceFormats != null && !deviceFormats.holds(layout.cacheTypes()))
			throw new KvSessionFileException(Reason.REFUSED, handler + ": the device KV keeps FP16 beside the "
					+ layout.cacheTypes().typeK().cliName() + "/" + layout.cacheTypes().typeV().cliName()
					+ " host KV, so rows rebuilt from a saved file would differ from the session's own;"
					+ " restore it where GPU attention is off or the device KV holds the host type");
	}

	/**
	 * Reads {@code file} into {@code layers} new K and V layers of {@code layout} for {@code requestId}. Nothing is
	 * installed; on failure the new layers are released and the exception passed on.
	 */
	static Restored read(SessionKvLayout layout, int layers, String requestId, KvSessionStamp stamp, Path file)
			throws IOException {
		SessionKvTensor[] k = layout.newKLayers(requestId, layers);
		SessionKvTensor[] v = layout.newVLayers(requestId, layers);
		try {
			long bytes = KvSessionFile.read(file, stamp, k, v);
			return new Restored(k, v, bytes);
		} catch (IOException | RuntimeException | Error e) {
			SessionKvLayout.releaseLayers(k);
			SessionKvLayout.releaseLayers(v);
			throw e;
		}
	}

	/** Puts the restored layers under {@code requestId}, releasing what it held. */
	static void install(Map<String, SessionKvTensor[]> kvCacheK, Map<String, SessionKvTensor[]> kvCacheV,
			String requestId, Restored r) {
		SessionKvLayout.releaseLayers(kvCacheK.put(requestId, r.k()));
		SessionKvLayout.releaseLayers(kvCacheV.put(requestId, r.v()));
	}
}
