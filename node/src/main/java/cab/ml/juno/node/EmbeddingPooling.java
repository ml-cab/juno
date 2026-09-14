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
 * Reduces a per-position hidden-state matrix — as returned by
 * {@link LocalInferencePipeline#embedTokens} — into a single embedding
 * vector, per {@link PoolingMode}.
 */
public final class EmbeddingPooling {

	private EmbeddingPooling() {
	}

	/**
	 * @param hidden one RMS/LayerNorm-normalized hidden vector per prompt
	 *               position, {@code hidden[pos][dim]}; every row must have the
	 *               same length
	 * @param mode   reduction strategy
	 * @return a new {@code float[hiddenDim]}, independent of {@code hidden}
	 * @throws IllegalArgumentException if {@code hidden} is null or empty
	 */
	public static float[] pool(float[][] hidden, PoolingMode mode) {
		if (hidden == null || hidden.length == 0)
			throw new IllegalArgumentException("hidden must not be empty");
		return switch (mode) {
		case LAST -> hidden[hidden.length - 1].clone();
		case CLS -> hidden[0].clone();
		case MEAN -> mean(hidden);
		};
	}

	private static float[] mean(float[][] hidden) {
		int dim = hidden[0].length;
		float[] sum = new float[dim];
		for (float[] row : hidden) {
			if (row.length != dim)
				throw new IllegalArgumentException("all hidden rows must share the same dimension");
			for (int i = 0; i < dim; i++)
				sum[i] += row[i];
		}
		for (int i = 0; i < dim; i++)
			sum[i] /= hidden.length;
		return sum;
	}
}
