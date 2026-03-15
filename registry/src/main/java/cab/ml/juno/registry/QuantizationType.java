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
package cab.ml.juno.registry;

/**
 * Model weight quantization format.
 *
 * Each type carries its effective bytes-per-parameter, used by ModelDescriptor
 * to estimate VRAM requirements per layer without needing to read the actual
 * weight file.
 *
 * This is also the foundation for quantization-aware sharding: nodes with less
 * VRAM can participate using lower-precision shards (e.g. INT4), while
 * high-VRAM nodes run FP16 for better quality.
 *
 * Bytes-per-param approximations: FP32 4.0 — full precision (training) BF16 2.0
 * — brain float16 (A100/H100 native) FP16 2.0 — half precision (most GPU
 * inference) INT8 1.0 — 8-bit quantized INT4 0.5 — 4-bit quantized Q8_0 1.0 —
 * GGUF 8-bit block quantization Q4_K_M 0.5 — GGUF 4-bit k-quant, medium
 * (recommended default) Q4_0 0.5 — GGUF 4-bit legacy Q5_K_M 0.625 — GGUF 5-bit
 * k-quant, medium (quality/size balance) Q6_K 0.75 — GGUF 6-bit k-quant (near
 * FP16 quality)
 */
public enum QuantizationType {

	FP32("FP32", 4.000), BF16("BF16", 2.000), FP16("FP16", 2.000), INT8("INT8", 1.000), INT4("INT4", 0.500),
	Q8_0("Q8_0", 1.000), Q4_K_M("Q4_K_M", 0.500), Q4_0("Q4_0", 0.500), Q5_K_M("Q5_K_M", 0.625), Q6_K("Q6_K", 0.750);

	private final String displayName;
	private final double bytesPerParam;

	QuantizationType(String displayName, double bytesPerParam) {
		this.displayName = displayName;
		this.bytesPerParam = bytesPerParam;
	}

	/** Human-readable name used in logs and API responses. */
	public String displayName() {
		return displayName;
	}

	/**
	 * Effective bytes per model parameter. Used to estimate VRAM per layer:
	 * params_per_layer × bytesPerParam.
	 */
	public double bytesPerParam() {
		return bytesPerParam;
	}
}
