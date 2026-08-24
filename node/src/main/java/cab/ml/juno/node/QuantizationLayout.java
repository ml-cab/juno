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
 * Layout metadata for a GGUF / GGML quantisation type used by Tier-5 codecs.
 *
 * <p>K-quants share a 256-element super-block ({@link #QK_K}). Q4_K / Q5_K are
 * affine (scale + min); Q6_K is symmetric (scaled, no additive zero/min).
 */
public record QuantizationLayout(
		int typeId,
		String name,
		int blockWidth,
		int subBlockWidth,
		int blockBytes,
		boolean affine,
		boolean symmetric) {

	/** llama.cpp / GGUF K-quant super-block width. */
	public static final int QK_K = 256;

	public static final int TYPE_Q4_K = 12;
	public static final int TYPE_Q5_K = 13;
	public static final int TYPE_Q6_K = 14;

	public static final QuantizationLayout Q4_K = new QuantizationLayout(
			TYPE_Q4_K, "Q4_K", QK_K, 32, 144, true, false);

	public static final QuantizationLayout Q5_K = new QuantizationLayout(
			TYPE_Q5_K, "Q5_K", QK_K, 32, 176, true, false);

	public static final QuantizationLayout Q6_K = new QuantizationLayout(
			TYPE_Q6_K, "Q6_K", QK_K, 16, 210, false, true);

	/**
	 * @param typeId GGML type ID
	 * @return layout for a supported K-quant, or {@code null} if unsupported
	 */
	public static QuantizationLayout forType(int typeId) {
		return switch (typeId) {
			case TYPE_Q4_K -> Q4_K;
			case TYPE_Q5_K -> Q5_K;
			case TYPE_Q6_K -> Q6_K;
			default -> null;
		};
	}

	/**
	 * @throws IllegalArgumentException if {@code typeId} is not a Tier-5 K-quant
	 */
	public static QuantizationLayout require(int typeId) {
		QuantizationLayout layout = forType(typeId);
		if (layout == null) {
			throw new IllegalArgumentException("Unsupported GGML K-quant type " + typeId);
		}
		return layout;
	}

	/** Encoded byte length for {@code nelems} logical elements. */
	public long encodedBytes(long nelems) {
		validateElementCount(nelems);
		return (nelems / blockWidth) * (long) blockBytes;
	}

	/**
	 * Rejects non-positive, overflowed, or non-block-aligned element counts.
	 *
	 * @throws IllegalArgumentException on malformed dimensions
	 */
	public void validateElementCount(long nelems) {
		if (nelems <= 0) {
			throw new IllegalArgumentException(name + ": nelems must be positive, got " + nelems);
		}
		if (nelems % blockWidth != 0) {
			throw new IllegalArgumentException(
					name + ": nelems=" + nelems + " not divisible by blockWidth=" + blockWidth);
		}
		long blocks = nelems / blockWidth;
		if (blocks > Integer.MAX_VALUE / blockBytes) {
			throw new IllegalArgumentException(name + ": encoded length overflows int for nelems=" + nelems);
		}
	}

	/**
	 * Validates a row-major matrix whose columns form complete K-quant blocks.
	 *
	 * @throws IllegalArgumentException on malformed dimensions
	 */
	public void validateMatrix(int rows, int cols) {
		if (rows <= 0 || cols <= 0) {
			throw new IllegalArgumentException(
					name + ": rows and cols must be positive, got " + rows + "x" + cols);
		}
		if (cols % blockWidth != 0) {
			throw new IllegalArgumentException(
					name + ": cols=" + cols + " not divisible by blockWidth=" + blockWidth);
		}
		validateElementCount((long) rows * cols);
	}
}
