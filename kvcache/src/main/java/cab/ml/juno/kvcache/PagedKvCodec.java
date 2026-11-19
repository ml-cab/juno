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
package cab.ml.juno.kvcache;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Serialize / restore paged (or dense-equivalent) K/V into {@link KVBlock} payloads.
 *
 * <p>Layout matches {@code NodeKVCacheAdapter}: F32+F32 is the headerless float32 LE layout
 * (K then V); otherwise {@code 'K''V' ver=1} + type nibble + packed payloads.
 * f16 rows are their halves, q8_0 and q4_0 rows their GGUF blocks ({@link KvRowCodec}).
 */
public final class PagedKvCodec {

	private static final byte MAGIC0 = 'K';
	private static final byte MAGIC1 = 'V';
	private static final byte VERSION = 1;

	private PagedKvCodec() {
	}

	public static byte[] encode(PagedKvTensor k, PagedKvTensor v, int seqLen) {
		if (k == null || v == null)
			throw new IllegalArgumentException("k and v must not be null");
		if (k.kvDim() != v.kvDim())
			throw new IllegalArgumentException("k/v kvDim mismatch");
		float[] kF = k.toFloatArray(seqLen);
		float[] vF = v.toFloatArray(seqLen);
		return encodeFloats(kF, vF, seqLen, k.kvDim(), k.type(), v.type());
	}

	public static byte[] encodeFloats(float[] k, float[] v, int seqLen, int kvDim,
			KvElementType kType, KvElementType vType) {
		if (kType.storesFloats() && vType.storesFloats()) {
			int floatsPerSeq = seqLen * kvDim;
			byte[] data = new byte[floatsPerSeq * Float.BYTES * 2];
			ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < floatsPerSeq; i++)
				bb.putFloat(k[i]);
			for (int i = 0; i < floatsPerSeq; i++)
				bb.putFloat(v[i]);
			return data;
		}
		int kBytes = seqLen * bytesPerToken(kType, kvDim);
		int vBytes = seqLen * bytesPerToken(vType, kvDim);
		byte[] data = new byte[4 + kBytes + vBytes];
		data[0] = MAGIC0;
		data[1] = MAGIC1;
		data[2] = VERSION;
		data[3] = (byte) ((kType.ordinal() << 4) | (vType.ordinal() & 0x0f));
		encodePayload(k, seqLen, kvDim, kType, data, 4);
		encodePayload(v, seqLen, kvDim, vType, data, 4 + kBytes);
		return data;
	}

	public static void decodeInto(KVBlock blk, int kvDim, PagedKvTensor k, PagedKvTensor v) {
		if (blk == null || k == null || v == null)
			throw new IllegalArgumentException("blk/k/v must not be null");
		int seqLen = blk.sequenceLen();
		float[][] pair = decodeFloats(blk, kvDim);
		k.loadFloatPrefix(pair[0], seqLen);
		v.loadFloatPrefix(pair[1], seqLen);
	}

	/** Returns {@code {kFloats, vFloats}}. */
	public static float[][] decodeFloats(KVBlock blk, int kvDim) {
		int seqLen = blk.sequenceLen();
		byte[] data = blk.data();
		if (isVersioned(data)) {
			KvElementType kType = typeFromNibble(data[3] >> 4);
			KvElementType vType = typeFromNibble(data[3] & 0x0f);
			int kBytes = seqLen * bytesPerToken(kType, kvDim);
			float[] k = decodePayload(data, 4, seqLen, kvDim, kType);
			float[] v = decodePayload(data, 4 + kBytes, seqLen, kvDim, vType);
			return new float[][] { k, v };
		}
		int floatsPerSeq = seqLen * kvDim;
		float[] k = new float[floatsPerSeq];
		float[] v = new float[floatsPerSeq];
		ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < floatsPerSeq; i++)
			k[i] = bb.getFloat();
		for (int i = 0; i < floatsPerSeq; i++)
			v[i] = bb.getFloat();
		return new float[][] { k, v };
	}

	private static void encodePayload(float[] src, int seqLen, int kvDim, KvElementType type,
			byte[] dst, int dstOff) {
		switch (type) {
		case F32 -> {
			ByteBuffer bb = ByteBuffer.wrap(dst, dstOff, seqLen * kvDim * Float.BYTES)
					.order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < seqLen * kvDim; i++)
				bb.putFloat(src[i]);
		}
		case F16, Q8_0, Q4_0 -> {
			int bpt = KvRowCodec.rowBytes(type, kvDim);
			for (int p = 0; p < seqLen; p++)
				KvRowCodec.encode(type, src, p * kvDim, kvDim, dst, dstOff + p * bpt);
		}
		}
	}

	private static float[] decodePayload(byte[] data, int off, int seqLen, int kvDim, KvElementType type) {
		float[] out = new float[seqLen * kvDim];
		switch (type) {
		case F32 -> {
			ByteBuffer bb = ByteBuffer.wrap(data, off, seqLen * kvDim * Float.BYTES)
					.order(ByteOrder.LITTLE_ENDIAN);
			for (int i = 0; i < out.length; i++)
				out[i] = bb.getFloat();
		}
		case F16, Q8_0, Q4_0 -> {
			int bpt = KvRowCodec.rowBytes(type, kvDim);
			for (int p = 0; p < seqLen; p++)
				KvRowCodec.decode(type, data, off + p * bpt, out, p * kvDim, kvDim);
		}
		}
		return out;
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
		return KvRowCodec.rowBytes(type, kvDim);
	}
}
