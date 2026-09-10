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

import java.util.Locale;

/**
 * In-process / serialized KV element type for {@code --cache-type-k/v}.
 *
 * <p>{@link #F16} is the CLI name for the current float32 path (bit-compatible
 * with pre–quantized-KV behavior). It does <em>not</em> pack IEEE half floats.
 */
public enum KvElementType {
	F16,
	Q8_0;

	public static KvElementType parse(String spec) {
		if (spec == null || spec.isBlank())
			return F16;
		String s = spec.strip().toLowerCase(Locale.ROOT);
		return switch (s) {
		case "f16", "fp16", "float", "f32", "fp32" -> F16;
		case "q8_0", "q8" -> Q8_0;
		default -> throw new IllegalArgumentException(
				"cache type must be f16|q8_0 (got " + spec + ")");
		};
	}

	public String cliName() {
		return switch (this) {
		case F16 -> "f16";
		case Q8_0 -> "q8_0";
		};
	}

	/** Persistent bytes per float element (q8_0 uses GGUF block packing). */
	public double bytesPerElement() {
		return switch (this) {
		case F16 -> Float.BYTES;
		case Q8_0 -> (double) Q8_0KvCodec.BLOCK_BYTES / Q8_0KvCodec.QK;
		};
	}
}
