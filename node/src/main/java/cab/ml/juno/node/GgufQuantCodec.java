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

/**
 * Shared GGUF quantisation codec facade for Tier-5 merge and training paths.
 *
 * <p>K-quant (Q4_K / Q5_K / Q6_K) work delegates to {@link GgufKQuantCodec}.
 * Other GGML types remain on their existing call sites until extracted.
 *
 * <p>Thread-safe: all methods are static and stateless.
 */
public final class GgufQuantCodec {

	private GgufQuantCodec() {}

	/** @return encoder strategy id for the given type, or {@code null} if unsupported here */
	public static String encoderId(int typeId) {
		return QuantizationLayout.forType(typeId) != null ? GgufKQuantCodec.ENCODER_ID : null;
	}

	public static QuantizationLayout layout(int typeId) {
		return QuantizationLayout.require(typeId);
	}

	public static float[] decode(byte[] raw, int typeId) {
		return GgufKQuantCodec.decode(raw, typeId);
	}

	public static float[] decodeRows(byte[] raw, int typeId, int rows, int cols) {
		return GgufKQuantCodec.decodeRows(raw, typeId, rows, cols);
	}

	public static byte[] encode(float[] data, int typeId) {
		return GgufKQuantCodec.encode(data, typeId);
	}

	/**
	 * Byte-preserving no-op: copy raw tensor bytes without decode/re-encode.
	 * This is the only path that guarantees byte-identical output for a zero delta.
	 */
	public static byte[] copyRawUnchanged(byte[] raw) {
		return GgufKQuantCodec.copyRawUnchanged(raw);
	}

	/**
	 * Decode → encode reconstruction metrics for a K-quant tensor.
	 */
	public static QuantizedMergeMetrics reconstructionMetrics(byte[] raw, int typeId) {
		float[] decoded = decode(raw, typeId);
		byte[] reencoded = encode(decoded, typeId);
		float[] roundTrip = decode(reencoded, typeId);
		return QuantizedMergeMetrics.ofReconstruction(decoded, roundTrip);
	}
}
