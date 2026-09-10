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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.util.Optional;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.KVBlock;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.KVKey;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.PagedKvCodec;
import cab.ml.juno.kvcache.PagedKvTensor;
import cab.ml.juno.kvcache.Q8_0KvCodec;

/**
 * Bridges the transformer handler's in-process KV tensors and the
 * {@link KVCacheManager} (GPU + CPU tiers with LRU eviction).
 *
 * <h3>KVBlock serialisation</h3>
 * <ul>
 * <li>{@link KvElementType#F16} + {@link KvElementType#F16} (default): legacy
 * float32 LE — K then V.</li>
 * <li>Otherwise: {@code 'K''V' ver=1 kType vType} then per-token packed payloads.</li>
 * </ul>
 */
public final class NodeKVCacheAdapter {

	private static final Logger log = Logger.getLogger(NodeKVCacheAdapter.class.getName());

	private static final byte MAGIC0 = 'K';
	private static final byte MAGIC1 = 'V';
	private static final byte VERSION = 1;

	private final KVCacheManager manager;

	public NodeKVCacheAdapter(KVCacheManager manager) {
		if (manager == null)
			throw new IllegalArgumentException("manager must not be null");
		this.manager = manager;
	}

	/**
	 * Serialise float K/V (legacy F16 path) into the manager.
	 */
	public void flush(String requestId, int absoluteLayerIndex,
			float[] kData, float[] vData,
			int seqLen, int kvDim) {
		flush(requestId, absoluteLayerIndex, kData, vData, seqLen, kvDim,
				KvElementType.F16, KvElementType.F16);
	}

	public void flush(String requestId, int absoluteLayerIndex,
			float[] kData, float[] vData,
			int seqLen, int kvDim,
			KvElementType kType, KvElementType vType) {
		if (kType == KvElementType.F16 && vType == KvElementType.F16) {
			int floatsPerSeq = seqLen * kvDim;
			int bytesPerSeq = floatsPerSeq * Float.BYTES;
			byte[] data = new byte[bytesPerSeq * 2];
			ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < floatsPerSeq; i++)
				bb.putFloat(kData[i]);
			for (int i = 0; i < floatsPerSeq; i++)
				bb.putFloat(vData[i]);
			putBlock(requestId, absoluteLayerIndex, data, seqLen, kType, vType);
			return;
		}
		DenseKvTensor kT = new DenseKvTensor(kType, kvDim, seqLen);
		DenseKvTensor vT = new DenseKvTensor(vType, kvDim, seqLen);
		kT.loadFloatPrefix(kData, seqLen);
		vT.loadFloatPrefix(vData, seqLen);
		flush(requestId, absoluteLayerIndex, kT, vT, seqLen);
	}

	/** Write-through from in-process tensors (preferred). */
	public void flush(String requestId, int absoluteLayerIndex,
			DenseKvTensor k, DenseKvTensor v, int seqLen) {
		KvElementType kType = k.type();
		KvElementType vType = v.type();
		if (kType == KvElementType.F16 && vType == KvElementType.F16) {
			flush(requestId, absoluteLayerIndex, k.toFloatArray(seqLen), v.toFloatArray(seqLen),
					seqLen, k.kvDim(), kType, vType);
			return;
		}
		int kBytes = seqLen * k.bytesPerToken();
		int vBytes = seqLen * v.bytesPerToken();
		byte[] data = new byte[4 + kBytes + vBytes];
		data[0] = MAGIC0;
		data[1] = MAGIC1;
		data[2] = VERSION;
		data[3] = (byte) ((kType.ordinal() << 4) | (vType.ordinal() & 0x0f));
		encodeTensorPayload(k, seqLen, data, 4);
		encodeTensorPayload(v, seqLen, data, 4 + kBytes);
		putBlock(requestId, absoluteLayerIndex, data, seqLen, kType, vType);
	}

	/** Write-through from paged tensors (continuous schedule). */
	public void flush(String requestId, int absoluteLayerIndex,
			PagedKvTensor k, PagedKvTensor v, int seqLen) {
		byte[] data = PagedKvCodec.encode(k, v, seqLen);
		putBlock(requestId, absoluteLayerIndex, data, seqLen, k.type(), v.type());
	}

	/**
	 * Restore into empty paged tensors when the manager holds a typed blob.
	 *
	 * @return true if restored
	 */
	public boolean tryRestorePaged(String requestId, int absoluteLayerIndex, int kvDim,
			PagedKvTensor k, PagedKvTensor v) {
		Optional<KVBlock> blk = manager.get(new KVKey(requestId, absoluteLayerIndex));
		if (blk.isEmpty())
			return false;
		PagedKvCodec.decodeInto(blk.get(), kvDim, k, v);
		log.fine("KV restored (paged) from manager: requestId=" + requestId
				+ " layer=" + absoluteLayerIndex + " seqLen=" + blk.get().sequenceLen());
		return true;
	}

	public Optional<KvPair> tryRestore(String requestId, int absoluteLayerIndex, int kvDim) {
		KVKey key = new KVKey(requestId, absoluteLayerIndex);
		return manager.get(key).map(blk -> decodeBlock(blk, kvDim));
	}

	public void evict(String requestId) {
		manager.evict(requestId);
		log.fine("KV evicted from manager: requestId=" + requestId);
	}

	public KVCacheManager manager() {
		return manager;
	}

	private void putBlock(String requestId, int layer, byte[] data, int seqLen,
			KvElementType kType, KvElementType vType) {
		KVKey key = new KVKey(requestId, layer);
		Instant now = Instant.now();
		manager.put(key, new KVBlock(key, data, seqLen, layer, now, now, kType, vType));
	}

	private static KvPair decodeBlock(KVBlock blk, int kvDim) {
		int seqLen = blk.sequenceLen();
		byte[] data = blk.data();
		if (isVersioned(data)) {
			KvElementType kType = typeFromNibble(data[3] >> 4);
			KvElementType vType = typeFromNibble(data[3] & 0x0f);
			int kBytes = seqLen * bytesPerToken(kType, kvDim);
			float[] k = decodePayload(data, 4, seqLen, kvDim, kType);
			float[] v = decodePayload(data, 4 + kBytes, seqLen, kvDim, vType);
			return new KvPair(k, v);
		}
		// Legacy float32
		int floatsPerSeq = seqLen * kvDim;
		float[] k = new float[floatsPerSeq];
		float[] v = new float[floatsPerSeq];
		ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < floatsPerSeq; i++)
			k[i] = bb.getFloat();
		for (int i = 0; i < floatsPerSeq; i++)
			v[i] = bb.getFloat();
		log.fine("KV restored from manager: requestId=" + blk.key().requestId()
				+ " layer=" + blk.layerIndex() + " seqLen=" + seqLen);
		return new KvPair(k, v);
	}

	private static boolean isVersioned(byte[] data) {
		return data.length >= 4 && data[0] == MAGIC0 && data[1] == MAGIC1 && data[2] == VERSION;
	}

	private static KvElementType typeFromNibble(int n) {
		KvElementType[] vals = KvElementType.values();
		int i = n & 0x0f;
		if (i < 0 || i >= vals.length)
			throw new IllegalArgumentException("unknown KV element type nibble " + i);
		return vals[i];
	}

	private static int bytesPerToken(KvElementType type, int kvDim) {
		return switch (type) {
		case F16 -> kvDim * Float.BYTES;
		case Q8_0 -> Q8_0KvCodec.encodedBytes(kvDim);
		};
	}

	private static void encodeTensorPayload(DenseKvTensor t, int seqLen, byte[] dst, int dstOff) {
		float[] floats = t.toFloatArray(seqLen);
		switch (t.type()) {
		case F16 -> {
			ByteBuffer bb = ByteBuffer.wrap(dst, dstOff, seqLen * t.kvDim() * Float.BYTES)
					.order(ByteOrder.LITTLE_ENDIAN);
			for (float f : floats)
				bb.putFloat(f);
		}
		case Q8_0 -> {
			int bpt = t.bytesPerToken();
			for (int p = 0; p < seqLen; p++)
				Q8_0KvCodec.encode(floats, p * t.kvDim(), t.kvDim(), dst, dstOff + p * bpt);
		}
		}
	}

	private static float[] decodePayload(byte[] data, int off, int seqLen, int kvDim, KvElementType type) {
		float[] out = new float[seqLen * kvDim];
		switch (type) {
		case F16 -> {
			ByteBuffer bb = ByteBuffer.wrap(data, off, seqLen * kvDim * Float.BYTES)
					.order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < out.length; i++)
				out[i] = bb.getFloat();
		}
		case Q8_0 -> {
			int bpt = Q8_0KvCodec.encodedBytes(kvDim);
			for (int p = 0; p < seqLen; p++)
				Q8_0KvCodec.decode(data, off + p * bpt, out, p * kvDim, kvDim);
		}
		}
		return out;
	}

	public record KvPair(float[] k, float[] v) {
	}
}
