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
 * How {@link EmbeddingPooling} reduces a per-position hidden-state matrix
 * (one RMS/LayerNorm-normalized vector per prompt token) into a single
 * embedding vector.
 */
public enum PoolingMode {

	/** Average of the hidden vector across every prompt position. */
	MEAN,

	/** Hidden vector at the first prompt position. */
	CLS,

	/**
	 * Hidden vector at the last prompt position. Cheapest option, but on
	 * chat-tuned (not embedding-trained) models it tends to over-weight the
	 * final tokens and under-represent the rest of the prompt.
	 */
	LAST;

	/**
	 * @param value {@code "mean"}, {@code "cls"}, or {@code "last"} (case
	 *              insensitive); {@code null} defaults to {@link #MEAN}
	 * @throws IllegalArgumentException on any other value
	 */
	public static PoolingMode parse(String value) {
		if (value == null)
			return MEAN;
		return switch (value.strip().toLowerCase()) {
		case "mean" -> MEAN;
		case "cls" -> CLS;
		case "last" -> LAST;
		default -> throw new IllegalArgumentException("Unknown pooling mode '" + value + "' (expected mean|cls|last)");
		};
	}
}
