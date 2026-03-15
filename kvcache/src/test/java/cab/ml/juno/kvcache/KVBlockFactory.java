package cab.ml.juno.kvcache;

import java.time.Instant;

import cab.ml.juno.kvcache.KVBlock;
import cab.ml.juno.kvcache.KVKey;

/** Test helper — builds KVBlock instances without boilerplate. */
final class KVBlockFactory {

	static KVBlock block(String requestId, int layer, int sizeBytes) {
		KVKey key = new KVKey(requestId, layer);
		byte[] data = new byte[sizeBytes];
		return new KVBlock(key, data, 100, layer, Instant.now(), Instant.now());
	}

	static KVBlock block(String requestId, int layer) {
		return block(requestId, layer, 1024);
	}
}
