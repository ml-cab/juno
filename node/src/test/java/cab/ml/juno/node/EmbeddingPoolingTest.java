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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class EmbeddingPoolingTest {

	private static final float[][] FIXTURE = { { 1f, 2f, 3f }, { 3f, 4f, 5f }, { 5f, 6f, 7f } };

	@Test
	void mean_averages_every_position() {
		float[] result = EmbeddingPooling.pool(FIXTURE, PoolingMode.MEAN);
		assertThat(result).containsExactly(3f, 4f, 5f);
	}

	@Test
	void cls_returns_first_position() {
		float[] result = EmbeddingPooling.pool(FIXTURE, PoolingMode.CLS);
		assertThat(result).containsExactly(1f, 2f, 3f);
	}

	@Test
	void last_returns_final_position() {
		float[] result = EmbeddingPooling.pool(FIXTURE, PoolingMode.LAST);
		assertThat(result).containsExactly(5f, 6f, 7f);
	}

	@Test
	void single_position_is_identical_for_every_mode() {
		float[][] single = { { 9f, 8f, 7f } };
		assertThat(EmbeddingPooling.pool(single, PoolingMode.MEAN)).containsExactly(9f, 8f, 7f);
		assertThat(EmbeddingPooling.pool(single, PoolingMode.CLS)).containsExactly(9f, 8f, 7f);
		assertThat(EmbeddingPooling.pool(single, PoolingMode.LAST)).containsExactly(9f, 8f, 7f);
	}

	@Test
	void result_is_independent_of_input_array() {
		float[][] copy = { { 1f, 2f }, { 3f, 4f } };
		float[] result = EmbeddingPooling.pool(copy, PoolingMode.LAST);
		result[0] = 999f;
		assertThat(copy[1][0]).isEqualTo(3f);
	}

	@Test
	void rejects_empty_hidden_matrix() {
		assertThatThrownBy(() -> EmbeddingPooling.pool(new float[0][], PoolingMode.MEAN))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_null_hidden_matrix() {
		assertThatThrownBy(() -> EmbeddingPooling.pool(null, PoolingMode.MEAN))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_ragged_rows_for_mean() {
		float[][] ragged = { { 1f, 2f }, { 3f } };
		assertThatThrownBy(() -> EmbeddingPooling.pool(ragged, PoolingMode.MEAN))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void parse_accepts_known_values_case_insensitively() {
		assertThat(PoolingMode.parse("MEAN")).isEqualTo(PoolingMode.MEAN);
		assertThat(PoolingMode.parse("cls")).isEqualTo(PoolingMode.CLS);
		assertThat(PoolingMode.parse("Last")).isEqualTo(PoolingMode.LAST);
	}

	@Test
	void parse_defaults_to_mean_when_null() {
		assertThat(PoolingMode.parse(null)).isEqualTo(PoolingMode.MEAN);
	}

	@Test
	void parse_rejects_unknown_value() {
		assertThatThrownBy(() -> PoolingMode.parse("softmax")).isInstanceOf(IllegalArgumentException.class);
	}
}
