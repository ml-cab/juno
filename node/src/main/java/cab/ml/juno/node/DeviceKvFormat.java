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

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

import java.lang.foreign.MemorySegment;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRowCodec;

/**
 * How one side (the keys or the values) of a request's device KV ({@link DeviceKvCache}) is stored. The Java half of
 * {@code kv_format.cuh}: {@link #code} is the format number the kernels take, and {@link #rowBytes} is the row stride
 * they compute, so a row's address is defined once on each side of the launch.
 *
 * <ul>
 * <li>{@link #F16}: IEEE FP16, 2 bytes per value: the host {@code f16} KV's rows byte for byte ({@code F16KvCodec};
 * both round to nearest even). It also serves an {@code f32} host KV, whose values it holds rounded to FP16, and any
 * type the device cannot store.</li>
 * <li>{@link #Q8_0} and {@link #Q4_0}: the host codecs' blocks byte for byte ({@code Q8_0KvCodec},
 * {@code Q4_0KvCodec}: 34 and 18 bytes per 32 values), so the device holds exactly the values the host KV of that
 * type holds. They need a head width that is a multiple of 32: a row is whole blocks, every block starts at an even
 * byte, and a decode lane of the attention kernel reads whole blocks of one head.</li>
 * </ul>
 */
enum DeviceKvFormat {

	F16(0, KvElementType.F16), Q8_0(1, KvElementType.Q8_0), Q4_0(2, KvElementType.Q4_0);

	/** Values per quantized block. */
	static final int BLOCK = 32;

	/** {@code kv_format.cuh}'s {@code KV_F16}, {@code KV_Q8_0}, {@code KV_Q4_0}. */
	final int code;
	private final KvElementType hostType;

	DeviceKvFormat(int code, KvElementType hostType) {
		this.code = code;
		this.hostType = hostType;
	}

	/** The host KV type whose values this format holds. */
	KvElementType hostType() {
		return hostType;
	}

	boolean quantized() {
		return this != F16;
	}

	/**
	 * Whether the device can store {@code type}'s values for heads of {@code headDim} values: {@code f16} always, a
	 * quantized type for heads of whole blocks, {@code f32} never (the device copy has no float32 rows).
	 */
	static boolean canHold(KvElementType type, int headDim) {
		return type == KvElementType.F16 || (type.quantized() && headDim > 0 && headDim % BLOCK == 0);
	}

	/** The format holding {@code type}'s values for heads of {@code headDim}, or {@link #F16} where it cannot. */
	static DeviceKvFormat holding(KvElementType type, int headDim) {
		if (!canHold(type, headDim))
			return F16;
		return switch (type) {
		case F16, F32 -> F16;
		case Q8_0 -> Q8_0;
		case Q4_0 -> Q4_0;
		};
	}

	/** Device bytes of one row of {@code kvDim} values. */
	long rowBytes(int kvDim) {
		if (kvDim < 1)
			throw new IllegalArgumentException("kvDim must be >= 1 (got " + kvDim + ")");
		if (this == F16)
			return (long) kvDim * Short.BYTES;
		if (kvDim % BLOCK != 0)
			throw new IllegalArgumentException(name() + " device rows need kvDim to be a multiple of " + BLOCK
					+ " (got " + kvDim + ")");
		return KvRowCodec.rowBytes(hostType, kvDim);
	}

	/**
	 * Packs one row, {@code kvDim} floats from {@code src[srcOff..)}, into {@code dst} at {@code dstOff}: what the
	 * device kernels write for the same floats ({@code kv_encode_kv_rows}), for the host-staged writes.
	 */
	void pack(float[] src, int srcOff, int kvDim, MemorySegment dst, long dstOff) {
		if (this == F16) {
			for (int i = 0; i < kvDim; i++)
				dst.set(JAVA_SHORT_UNALIGNED, dstOff + 2L * i, Float.floatToFloat16(src[srcOff + i]));
			return;
		}
		byte[] row = new byte[(int) rowBytes(kvDim)];
		KvRowCodec.encode(hostType, src, srcOff, kvDim, row, 0);
		MemorySegment.copy(row, 0, dst, JAVA_BYTE, dstOff, row.length);
	}

	/** Unpacks one row at {@code src[srcOff..)} into {@code kvDim} floats at {@code dst[dstOff..)}. */
	void unpack(MemorySegment src, long srcOff, int kvDim, float[] dst, int dstOff) {
		if (this == F16) {
			for (int i = 0; i < kvDim; i++)
				dst[dstOff + i] = Float.float16ToFloat(src.get(JAVA_SHORT_UNALIGNED, srcOff + 2L * i));
			return;
		}
		byte[] row = new byte[(int) rowBytes(kvDim)];
		MemorySegment.copy(src, JAVA_BYTE, srcOff, row, 0, row.length);
		KvRowCodec.decode(hostType, row, 0, dst, dstOff, kvDim);
	}
}
