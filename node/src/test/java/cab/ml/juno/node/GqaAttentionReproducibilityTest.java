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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The attention kernel gives the same bits for the same inputs, launch after launch.
 *
 * <p>Each block reduces twice through one shared-memory array (the softmax max,
 * then the exponent sum). Without a barrier between the two, a warp that finishes
 * the second pass early overwrites the max before a slower warp has read it, and
 * that warp's share of the softmax is scaled by the wrong value. The result then
 * depends on warp timing: rare per block, but a 512-row prefill window launches
 * 512 x heads blocks per layer, and greedy output was seen to change between
 * identical requests. A wide window, launched many times, exposes it.
 */
@Tag("gpu")
@DisplayName("GPU attention kernel: bit-identical across launches")
class GqaAttentionReproducibilityTest {

	private static final int LAUNCHES = 100;
	private static final float TOL = 3e-3f;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(GqaAttentionKernel.isAvailable(), "Skipping - GPU attention kernel unavailable");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a 512-row prefill window gives identical output on every launch, and matches the CPU oracle")
	void prefillWindowIsReproducible() {
		int numHeads = 32, numKvHeads = 4, headDim = 64, window = 512;
		int gqaRatio = numHeads / numKvHeads;
		int kvDim = numKvHeads * headDim;
		Random rng = new Random(20261004L);

		DeviceKvCache kv = new DeviceKvCache(ctx, kvDim);
		try {
			for (int t = 0; t < window; t++)
				kv.appendToken(t, randomRow(rng, kvDim), randomRow(rng, kvDim));
			float[][] q = new float[window][];
			int[] seqLens = new int[window];
			DeviceKvCache[] kvPerRow = new DeviceKvCache[window];
			for (int b = 0; b < window; b++) {
				q[b] = randomRow(rng, numHeads * headDim);
				seqLens[b] = b + 1;
				kvPerRow[b] = kv;
			}

			CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
			float[][] first = null;
			int differing = 0;
			int firstDifferingLaunch = -1;
			for (int launch = 0; launch < LAUNCHES; launch++) {
				float[][] out = new float[window][];
				assertThat(attn.attendBatched(kvPerRow, q, seqLens, out, numHeads, headDim, gqaRatio, kvDim))
						.as("kernel must load on a CUDA-available host").isTrue();
				if (first == null) {
					first = out;
					continue;
				}
				if (!sameBits(first, out)) {
					differing++;
					if (firstDifferingLaunch < 0)
						firstDifferingLaunch = launch;
				}
			}
			assertThat(differing).as("launches (of %d) whose output differs from the first; first at launch %d",
					LAUNCHES - 1, firstDifferingLaunch).isZero();

			float[] kAll = kv.downloadK(window);
			float[] vAll = kv.downloadV(window);
			for (int b = 0; b < window; b += 37) {
				float[] cpu = new float[numHeads * headDim];
				GqaMath.attend(q[b], kAll, vAll, seqLens[b], cpu, new float[seqLens[b]], numHeads, headDim,
						gqaRatio, kvDim);
				for (int i = 0; i < cpu.length; i++)
					assertThat(first[b][i]).as("row %d [%d]", b, i).isCloseTo(cpu[i], within(TOL));
			}
		} finally {
			kv.close();
		}
	}

	private static boolean sameBits(float[][] a, float[][] b) {
		for (int r = 0; r < a.length; r++)
			for (int i = 0; i < a[r].length; i++)
				if (Float.floatToRawIntBits(a[r][i]) != Float.floatToRawIntBits(b[r][i]))
					return false;
		return true;
	}

	private static float[] randomRow(Random rng, int n) {
		float[] row = new float[n];
		for (int i = 0; i < n; i++)
			row[i] = rng.nextFloat() * 2f - 1f;
		return row;
	}
}
