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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * {@link KvElementType#F16} KV rows: each value's IEEE half-precision bits, two little-endian bytes, as
 * {@link Float#floatToFloat16} rounds it (to nearest even; overflow to infinity; gradual underflow), read back with
 * {@link Float#float16ToFloat}. The same bytes as an FP16 device KV row, so a host tensor can take a device copy's rows
 * as they are. A row has no blocks: {@link #decodeRange} reads any slice directly.
 */
public final class F16KvCodec {

	private static final VarHandle HALF = MethodHandles.byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN);

	private F16KvCodec() {
	}

	/** Stored size of {@code nFloats} values: two bytes each. */
	public static int encodedBytes(int nFloats) {
		if (nFloats < 0)
			throw new IllegalArgumentException("nFloats must be >= 0");
		return 2 * nFloats;
	}

	/** Stores {@code n} floats from {@code src[srcOff..)} as halves at {@code dst[dstOff..)}. */
	public static void encode(float[] src, int srcOff, int n, byte[] dst, int dstOff) {
		if (srcOff < 0 || n < 0 || srcOff + n > src.length)
			throw new IllegalArgumentException("range off=" + srcOff + " n=" + n + " len=" + src.length);
		if (dstOff < 0 || dstOff + 2 * n > dst.length)
			throw new IllegalArgumentException("dst too small for " + n + " values");
		for (int i = 0; i < n; i++)
			HALF.set(dst, dstOff + 2 * i, Float.floatToFloat16(src[srcOff + i]));
	}

	/** Reads {@code n} halves at {@code src[srcOff..)} into {@code dst[dstOff..)}. */
	public static void decode(byte[] src, int srcOff, float[] dst, int dstOff, int n) {
		if (dstOff < 0 || n < 0 || dstOff + n > dst.length)
			throw new IllegalArgumentException("range off=" + dstOff + " n=" + n + " len=" + dst.length);
		if (srcOff < 0 || srcOff + 2 * n > src.length)
			throw new IllegalArgumentException("src too small for " + n + " values");
		decodeRange(src, srcOff, 0, n, dst, dstOff);
	}

	/**
	 * Reads elements {@code [from, from + n)} of the row stored at {@code src[rowOff..)} into
	 * {@code dst[dstOff, dstOff + n)}: what {@link #decode} gives them, bit for bit. Attention calls this with one
	 * head's slice of a stored row. No range checks beyond the arrays' own: the caller is the attention loop.
	 */
	public static void decodeRange(byte[] src, int rowOff, int from, int n, float[] dst, int dstOff) {
		int so = rowOff + 2 * from;
		for (int i = 0; i < n; i++)
			dst[dstOff + i] = Float.float16ToFloat((short) HALF.get(src, so + 2 * i));
	}
}
