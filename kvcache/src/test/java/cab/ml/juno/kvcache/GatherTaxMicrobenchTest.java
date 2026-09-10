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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@DisplayName("GatherTaxMicrobench")
class GatherTaxMicrobenchTest {

	@Test
	@DisplayName("gqaInto is deterministic for fixed inputs")
	void gqa_deterministic() {
		int seq = 64;
		float[] q = new float[GatherTaxMicrobench.Q_DIM];
		float[] k = new float[seq * GatherTaxMicrobench.KV_DIM];
		float[] v = new float[seq * GatherTaxMicrobench.KV_DIM];
		for (int i = 0; i < q.length; i++)
			q[i] = (i % 7) * 0.01f;
		for (int i = 0; i < k.length; i++) {
			k[i] = (i % 11) * 0.001f;
			v[i] = (i % 13) * 0.001f;
		}
		float[] out1 = new float[GatherTaxMicrobench.Q_DIM];
		float[] out2 = new float[GatherTaxMicrobench.Q_DIM];
		float[] scores = new float[seq];
		GatherTaxMicrobench.gqaInto(q, k, v, seq, out1, scores);
		GatherTaxMicrobench.gqaInto(q, k, v, seq, out2, scores);
		for (int i = 0; i < out1.length; i++)
			assertThat(out2[i]).isCloseTo(out1[i], within(0f));
	}

	@Test
	@Timeout(120)
	@DisplayName("small cell produces finite timings and markdown")
	void small_cell_runs() {
		List<GatherTaxMicrobench.Cell> cells = GatherTaxMicrobench.runMatrix(
				new int[] { 256 }, new int[] { 1, 2 }, new int[] { 16 }, 1, 2);
		assertThat(cells).hasSize(2);
		for (GatherTaxMicrobench.Cell c : cells) {
			assertThat(c.denseAttnNs()).isPositive();
			assertThat(c.gatherNs()).isPositive();
			assertThat(c.pagedAttnNs()).isPositive();
			assertThat(c.gatherPctOfPaged()).isBetween(0.0, 100.0);
		}
		String md = GatherTaxMicrobench.formatReport(cells);
		assertThat(md).contains("Gather-tax microbench").contains("| 256 |");
	}
}
