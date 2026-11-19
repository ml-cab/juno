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
 * <p>{@link #F16} (the default) stores each value as an IEEE half, two bytes ({@link F16KvCodec}); {@link #F32}
 * stores the floats as they are, four bytes. {@link #Q8_0} and {@link #Q4_0} store each row in GGUF blocks of 32
 * values ({@link Q8_0KvCodec}, {@link Q4_0KvCodec}). {@link KvRowCodec} encodes and decodes every type but
 * {@code F32}, whose rows are float arrays ({@link #storesFloats}).
 *
 * <p>A serialized KV block names its types by ordinal, so a new type is appended,
 * never inserted.
 */
public enum KvElementType {
	F16,
	Q8_0,
	Q4_0,
	F32;

	public static KvElementType parse(String spec) {
		if (spec == null || spec.isBlank())
			return F16;
		String s = spec.strip().toLowerCase(Locale.ROOT);
		return switch (s) {
		case "f16", "fp16" -> F16;
		case "f32", "fp32", "float" -> F32;
		case "q8_0", "q8" -> Q8_0;
		case "q4_0", "q4" -> Q4_0;
		default -> throw new IllegalArgumentException(
				"cache type must be f16|f32|q8_0|q4_0 (got " + spec + ")");
		};
	}

	public String cliName() {
		return switch (this) {
		case F16 -> "f16";
		case Q8_0 -> "q8_0";
		case Q4_0 -> "q4_0";
		case F32 -> "f32";
		};
	}

	/** Rows of this type are float arrays read as they are ({@code F32}); every other type's rows are encoded bytes. */
	public boolean storesFloats() {
		return this == F32;
	}

	/** A block-quantized type: {@code Q8_0} or {@code Q4_0}. */
	public boolean quantized() {
		return this == Q8_0 || this == Q4_0;
	}

	/** Persistent bytes per float element (q8_0 and q4_0 use GGUF block packing). */
	public double bytesPerElement() {
		return switch (this) {
		case F16 -> Short.BYTES;
		case F32 -> Float.BYTES;
		case Q8_0 -> (double) Q8_0KvCodec.BLOCK_BYTES / Q8_0KvCodec.QK;
		case Q4_0 -> (double) Q4_0KvCodec.BLOCK_BYTES / Q4_0KvCodec.QK;
		};
	}
}
