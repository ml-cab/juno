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
package cab.ml.juno.lora;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

@DisplayName("QaLoraAdapter (Tier-5 Gate B)")
class QaLoraAdapterTest {

	private static final float FD_H = 1e-3f;
	private static final float FD_TOL = 2e-2f;

	private static LoraAdapterConfig cfg(int rank, float alpha) {
		return LoraAdapterConfig.of(rank, alpha, LoraScaling.STANDARD, LoraInitialization.KAIMING_UNIFORM,
				LoraMode.QA_LORA);
	}

	@Test
	@DisplayName("rejects inDim not divisible by groupWidth")
	void rejects_misaligned() {
		assertThatThrownBy(() -> new QaLoraAdapter(cfg(2, 2f), 10, 4, 4, new Random(1)))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("not divisible");
	}

	@Test
	@DisplayName("rejects non-QA mode config")
	void rejects_wrong_mode() {
		assertThatThrownBy(() -> new QaLoraAdapter(
				LoraAdapterConfig.of(2, 2f), 8, 4, 4, new Random(1)))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("QA_LORA");
	}

	@Test
	@DisplayName("sum pool is sum not average")
	void sum_not_average() {
		QaLoraAdapter a = QaLoraAdapter.fromWeights(cfg(1, 1f), 4, 1, 2,
				new float[] { 1f, 0f }, new float[] { 1f });
		float[] pooled = a.pool(new float[] { 1f, 1f, 2f, 2f });
		assertThat(pooled[0]).isEqualTo(2f);
		assertThat(pooled[1]).isEqualTo(4f);
	}

	@Test
	@DisplayName("forward equals dense expanded ΔW · x")
	void forward_matches_dense_oracle() {
		Random rng = new Random(7);
		QaLoraAdapter qa = new QaLoraAdapter(cfg(3, 6f), 8, 5, 2, rng);
		// Non-zero B
		for (int i = 0; i < qa.b().length; i++)
			qa.b()[i] = (rng.nextFloat() * 2 - 1) * 0.1f;

		float[] x = new float[8];
		for (int i = 0; i < x.length; i++)
			x[i] = rng.nextFloat();

		float[] delta = qa.forward(x);
		float[] dense = qa.expandDenseDelta();
		float[] oracle = new float[5];
		for (int o = 0; o < 5; o++) {
			float acc = 0f;
			for (int c = 0; c < 8; c++)
				acc += dense[o * 8 + c] * x[c];
			oracle[o] = acc;
		}
		assertThat(delta).usingComparatorWithPrecision(1e-5f).containsExactly(oracle);
	}

	@Test
	@DisplayName("finite differences for A, B, and input")
	void finite_differences() {
		Random rng = new Random(11);
		QaLoraAdapter qa = new QaLoraAdapter(cfg(2, 4f), 8, 4, 4, rng);
		for (int i = 0; i < qa.b().length; i++)
			qa.b()[i] = (rng.nextFloat() * 2 - 1) * 0.05f;

		float[] x = new float[8];
		for (int i = 0; i < 8; i++)
			x[i] = rng.nextFloat();
		float[] upstream = new float[4];
		for (int i = 0; i < 4; i++)
			upstream[i] = rng.nextFloat() * 2 - 1;

		qa.zeroGrad();
		float[] gradX = qa.backward(upstream, x);

		// grad A
		for (int i = 0; i < qa.a().length; i++) {
			float orig = qa.a()[i];
			qa.a()[i] = orig + FD_H;
			float[] yp = qa.forward(x);
			qa.a()[i] = orig - FD_H;
			float[] ym = qa.forward(x);
			qa.a()[i] = orig;
			float fd = 0f;
			for (int o = 0; o < 4; o++)
				fd += upstream[o] * (yp[o] - ym[o]) / (2f * FD_H);
			assertThat(qa.gradA()[i]).as("gradA[%d]", i).isCloseTo(fd, within(FD_TOL));
		}

		// grad B
		for (int i = 0; i < qa.b().length; i++) {
			float orig = qa.b()[i];
			qa.b()[i] = orig + FD_H;
			float[] yp = qa.forward(x);
			qa.b()[i] = orig - FD_H;
			float[] ym = qa.forward(x);
			qa.b()[i] = orig;
			float fd = 0f;
			for (int o = 0; o < 4; o++)
				fd += upstream[o] * (yp[o] - ym[o]) / (2f * FD_H);
			assertThat(qa.gradB()[i]).as("gradB[%d]", i).isCloseTo(fd, within(FD_TOL));
		}

		// grad X
		for (int j = 0; j < 8; j++) {
			float orig = x[j];
			x[j] = orig + FD_H;
			float[] yp = qa.forward(x);
			x[j] = orig - FD_H;
			float[] ym = qa.forward(x);
			x[j] = orig;
			float fd = 0f;
			for (int o = 0; o < 4; o++)
				fd += upstream[o] * (yp[o] - ym[o]) / (2f * FD_H);
			assertThat(gradX[j]).as("gradX[%d]", j).isCloseTo(fd, within(FD_TOL));
		}
	}

	@Test
	@DisplayName("deterministic Kaiming init uses groupCount fan-in")
	void kaiming_uses_group_count() {
		QaLoraAdapter a = new QaLoraAdapter(cfg(2, 2f), 16, 4, 4, new Random(42));
		float bound = 1f / (float) Math.sqrt(4); // groupCount=4
		for (float v : a.a())
			assertThat(Math.abs(v)).isLessThanOrEqualTo(bound + 1e-6f);
	}
}
