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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Parity: {@link MatVec#sgemvSameX} matches serial {@link MatVec#sgemv} calls.
 */
@Tag("gpu")
@DisplayName("sgemvSameX shared-activation parity")
class SgemvSameXParityTest {

	@Test
	@DisplayName("FP16 sgemvSameX matches three serial sgemv")
	void fp16_sameX_matches_serial() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");

		Random rnd = new Random(11);
		int cols = 64;
		int[] rows = { 64, 32, 32 };
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = (rnd.nextFloat() * 2f) - 1f;

		try (GpuContext ctx = GpuContext.init(0)) {
			CudaMatVec mv = new CudaMatVec(ctx);
			assumeTrue(mv.supportsHalfResident(), "FP16 resident required");

			DeviceHalfMatrix[] mats = new DeviceHalfMatrix[3];
			float[][] expected = new float[3][];
			try {
				for (int i = 0; i < 3; i++) {
					float[] host = new float[rows[i] * cols];
					for (int j = 0; j < host.length; j++)
						host[j] = (rnd.nextFloat() * 2f) - 1f;
					mats[i] = mv.uploadHalf(host, rows[i], cols);
					expected[i] = mv.sgemv(mats[i], x);
				}
				float[][] got = mv.sgemvSameX(mats, x);
				assertThat(got.length).isEqualTo(3);
				for (int i = 0; i < 3; i++) {
					assertThat(got[i]).hasSize(rows[i]);
					for (int r = 0; r < rows[i]; r++)
						assertThat(got[i][r]).as("y[" + i + "][" + r + "]")
								.isCloseTo(expected[i][r], within(1e-3f));
				}
			} finally {
				for (DeviceHalfMatrix m : mats) {
					if (m != null && !m.isClosed())
						m.close();
				}
			}
		}
	}
}
