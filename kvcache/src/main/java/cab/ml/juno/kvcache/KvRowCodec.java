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

/**
 * Stored rows of a {@link KvElementType}: their size, and for the encoded types ({@link KvElementType#F16} halves,
 * {@link KvElementType#Q8_0} and {@link KvElementType#Q4_0} blocks) their encoding and decoding, in one place, so that
 * every KV store (dense, paged, serialized) and reader handles every type the same way. {@link KvElementType#F32} rows
 * are floats held as they are; the byte methods refuse it.
 */
public final class KvRowCodec {

	private KvRowCodec() {
	}

	/** Bytes one stored row of {@code kvDim} values takes: floats for {@code F32}, halves for {@code F16}, blocks otherwise. */
	public static int rowBytes(KvElementType type, int kvDim) {
		return switch (type) {
		case F32 -> kvDim * Float.BYTES;
		case F16 -> F16KvCodec.encodedBytes(kvDim);
		case Q8_0 -> Q8_0KvCodec.encodedBytes(kvDim);
		case Q4_0 -> Q4_0KvCodec.encodedBytes(kvDim);
		};
	}

	/** Encode {@code n} floats from {@code src[srcOff..)} as {@code type}'s blocks at {@code dst[dstOff..)}. */
	public static void encode(KvElementType type, float[] src, int srcOff, int n, byte[] dst, int dstOff) {
		switch (type) {
		case Q8_0 -> Q8_0KvCodec.encode(src, srcOff, n, dst, dstOff);
		case Q4_0 -> Q4_0KvCodec.encode(src, srcOff, n, dst, dstOff);
		case F16 -> F16KvCodec.encode(src, srcOff, n, dst, dstOff);
		case F32 -> throw new IllegalArgumentException(type.cliName() + " rows are stored as floats, not encoded");
		}
	}

	/** Decode {@code n} floats of {@code type}'s blocks at {@code src[srcOff..)} into {@code dst[dstOff..)}. */
	public static void decode(KvElementType type, byte[] src, int srcOff, float[] dst, int dstOff, int n) {
		switch (type) {
		case Q8_0 -> Q8_0KvCodec.decode(src, srcOff, dst, dstOff, n);
		case Q4_0 -> Q4_0KvCodec.decode(src, srcOff, dst, dstOff, n);
		case F16 -> F16KvCodec.decode(src, srcOff, dst, dstOff, n);
		case F32 -> throw new IllegalArgumentException(type.cliName() + " rows are stored as floats, not encoded");
		}
	}
}
