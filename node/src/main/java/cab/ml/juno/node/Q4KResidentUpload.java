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
 * Shared packed K-quant vs FP16-half upload for inference GPU residency.
 *
 * <p>When {@code tryMmq} is true and the tensor is a K-quant with a fused kernel
 * (Q4_K / Q5_K / Q6_K, see {@link DeviceQ4KMatrix#supportsType}), uploads packed
 * device bytes ({@link DeviceQ4KMatrix}). Otherwise dequantises and uploads
 * {@link DeviceHalfMatrix}. Used by Llama / Phi-3 / Qwen3 handlers so the fused
 * MMQ path stays consistent across architectures.
 */
public final class Q4KResidentUpload {

	private Q4KResidentUpload() {
	}

	/** True when packed K-quant upload should be preferred over FP16. */
	public static boolean preferPacked(boolean tryMmq, GgufReader.QuantizedTensor quant) {
		return tryMmq && quant != null && DeviceQ4KMatrix.supportsType(quant.type());
	}

	/**
	 * Upload a full projection matrix: packed K-quant when preferred, else FP16 half.
	 * Exactly one of {@link Slot#q4()} / {@link Slot#half()} is non-null on success.
	 */
	public static Slot upload(GpuMatVec cuda, GgufReader.QuantizedTensor quant, int rows, int cols,
			boolean tryMmq) {
		if (preferPacked(tryMmq, quant))
			return Slot.q4(cuda.uploadKQuant(quant.data(), rows, cols, quant.type()));
		return Slot.half(cuda.uploadHalf(LlamaTransformerHandler.dequantize(quant, rows, cols), rows, cols));
	}

	/**
	 * Write into layer slots: Q4 array when preferred, else half array. The unused
	 * slot at {@code li} stays null.
	 */
	public static void uploadInto(GpuMatVec cuda, GgufReader.QuantizedTensor quant, int rows, int cols,
			boolean tryMmq, int li, DeviceHalfMatrix[] halfSlot, DeviceQ4KMatrix[] q4Slot) {
		Slot s = upload(cuda, quant, rows, cols, tryMmq);
		if (s.q4() != null) {
			if (q4Slot == null)
				throw new IllegalStateException("Q4_K upload requested but q4Slot is null");
			q4Slot[li] = s.q4();
		} else {
			halfSlot[li] = s.half();
		}
	}

	public static void closeArray(DeviceQ4KMatrix[] a) {
		if (a == null)
			return;
		for (DeviceQ4KMatrix m : a) {
			if (m != null && !m.isClosed())
				m.close();
		}
	}

	public static void closeQuietly(DeviceQ4KMatrix m) {
		if (m != null && !m.isClosed())
			m.close();
	}

	/** Result of {@link #upload}: exactly one of q4 / half is non-null. */
	public record Slot(DeviceQ4KMatrix q4, DeviceHalfMatrix half) {
		static Slot q4(DeviceQ4KMatrix q4) {
			return new Slot(q4, null);
		}

		static Slot half(DeviceHalfMatrix half) {
			return new Slot(null, half);
		}

		void closeQuietly() {
			Q4KResidentUpload.closeQuietly(q4);
			if (half != null && !half.isClosed())
				half.close();
		}
	}
}
