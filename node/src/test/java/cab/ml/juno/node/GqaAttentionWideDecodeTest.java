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
 * The FP16 attention kernel at decode width (one row a block, each row over its own cache), where head widths up to
 * 128 run the wide decode block: 64 slots of 4 lanes, slot {@code s} taking keys {@code base + s, base + s + 64, ...},
 * merged over the 64 slots. Each case puts the rows where that geometry has edges: contexts on either side of a
 * multiple of 64 (slots with no key, a last stride that is partly empty), sliding windows whose first key is not on a
 * stride boundary, head widths that leave a lane's last quads empty (80, 96), and the 256-wide heads that keep the
 * 128-thread tiles. Every row is checked against the CPU oracle ({@link GqaMath#attend}); a row decoded among other
 * rows is held to its own launch bit for bit, the property the decode batch is built on.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=GqaAttentionWideDecodeTest}.
 */
@Tag("gpu")
@DisplayName("GPU attention at decode width: the 64-slot block over FP16 rows")
class GqaAttentionWideDecodeTest {

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

	private record Shape(String name, int numHeads, int numKvHeads, int headDim) {
		int gqaRatio() {
			return numHeads / numKvHeads;
		}

		int kvDim() {
			return numKvHeads * headDim;
		}

		int rowDim() {
			return numHeads * headDim;
		}
	}

	private static final Shape TINYLLAMA = new Shape("TinyLlama 32/4/64", 32, 4, 64);
	private static final Shape QWEN2 = new Shape("Qwen2.5-3B 16/2/128", 16, 2, 128);
	private static final Shape PHI3 = new Shape("Phi-3.5-mini 32/32/96", 32, 32, 96);
	private static final Shape PHI2 = new Shape("Phi-2 32/32/80", 32, 32, 80);
	private static final Shape MISTRAL = new Shape("Mistral 7B 32/8/128", 32, 8, 128);
	private static final Shape WIDE = new Shape("256-wide heads 8/2/256", 8, 2, 256);

	@Test
	@DisplayName("rows on either side of a 64-key stride match the oracle at every head width")
	void rowsAroundTheKeyStrideMatchTheOracle() {
		int[] seqLens = { 1, 2, 63, 64, 65, 127, 128, 129, 191, 1000 };
		for (Shape shape : new Shape[] { TINYLLAMA, QWEN2, PHI3, PHI2, MISTRAL, WIDE })
			runRows(shape, seqLens, 0);
	}

	@Test
	@DisplayName("a sliding window whose first key is off the stride matches the oracle")
	void slidingWindowsMatchTheOracle() {
		int[] seqLens = { 129, 300, 1000 };
		for (Shape shape : new Shape[] { TINYLLAMA, QWEN2, PHI3 })
			for (int window : new int[] { 1, 63, 64, 65, 100, 128 })
				runRows(shape, seqLens, window);
	}

	@Test
	@DisplayName("a row decoded among other rows is that row decoded alone, bit for bit")
	void aRowAmongOthersIsTheRowAlone() {
		int[] seqLens = { 1, 64, 65, 500, 2049 };
		for (Shape shape : new Shape[] { QWEN2, PHI3, WIDE }) {
			Random rng = new Random(shape.name().hashCode() + 5L);
			DeviceKvCache[] kv = caches(shape, rng, seqLens);
			try {
				float[][] q = randomRows(rng, seqLens.length, shape.rowDim());
				float[][] together = attend(shape, kv, q, seqLens, 0);
				for (int b = 0; b < seqLens.length; b++) {
					float[][] alone = attend(shape, new DeviceKvCache[] { kv[b] }, new float[][] { q[b] },
							new int[] { seqLens[b] }, 0);
					for (int i = 0; i < alone[0].length; i++)
						assertThat(Float.floatToRawIntBits(together[b][i]))
								.as("%s row %d (context %d) [%d]", shape.name(), b, seqLens[b], i)
								.isEqualTo(Float.floatToRawIntBits(alone[0][i]));
				}
			} finally {
				close(kv);
			}
		}
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	/** One decode launch: row {@code b} over its own cache of {@code seqLens[b]} keys; every row against the oracle. */
	private static void runRows(Shape shape, int[] seqLens, int window) {
		Random rng = new Random(shape.name().hashCode() * 31L + window);
		DeviceKvCache[] kv = caches(shape, rng, seqLens);
		try {
			float[][] q = randomRows(rng, seqLens.length, shape.rowDim());
			float[][] out = attend(shape, kv, q, seqLens, window);
			for (int b = 0; b < seqLens.length; b++) {
				int from = window > 0 && seqLens[b] > window ? seqLens[b] - window : 0;
				float[] expected = oracle(shape, q[b], kv[b].downloadK(seqLens[b]), kv[b].downloadV(seqLens[b]), from,
						seqLens[b]);
				assertClose(shape.name() + " context " + seqLens[b] + " window " + window, out[b], expected);
			}
		} finally {
			close(kv);
		}
	}

	private static DeviceKvCache[] caches(Shape shape, Random rng, int[] seqLens) {
		DeviceKvCache[] kv = new DeviceKvCache[seqLens.length];
		try {
			for (int b = 0; b < seqLens.length; b++) {
				kv[b] = new DeviceKvCache(ctx, shape.kvDim(), seqLens[b]);
				kv[b].appendWindow(0, randomRows(rng, seqLens[b], shape.kvDim()),
						randomRows(rng, seqLens[b], shape.kvDim()), seqLens[b]);
			}
			return kv;
		} catch (RuntimeException e) {
			close(kv);
			throw e;
		}
	}

	private static void close(DeviceKvCache[] kv) {
		for (DeviceKvCache k : kv)
			if (k != null)
				k.close();
	}

	private static float[][] attend(Shape shape, DeviceKvCache[] kv, float[][] q, int[] seqLens, int window) {
		CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
		try {
			float[][] out = new float[q.length][];
			boolean ok = attn.attendBatched(kv, q, seqLens, out, shape.numHeads(), shape.headDim(), shape.gqaRatio(),
					shape.kvDim(), window);
			assertThat(ok).as("kernel must load on a CUDA-available host").isTrue();
			return out;
		} finally {
			attn.close();
		}
	}

	/** The oracle over key rows {@code [from, seqLen)}. */
	private static float[] oracle(Shape shape, float[] q, float[] kHost, float[] vHost, int from, int seqLen) {
		int n = seqLen - from;
		int kvDim = shape.kvDim();
		float[] k = new float[n * kvDim];
		float[] v = new float[n * kvDim];
		System.arraycopy(kHost, from * kvDim, k, 0, n * kvDim);
		System.arraycopy(vHost, from * kvDim, v, 0, n * kvDim);
		float[] out = new float[shape.rowDim()];
		GqaMath.attend(q, k, v, n, out, new float[n], shape.numHeads(), shape.headDim(), shape.gqaRatio(), kvDim);
		return out;
	}

	private static void assertClose(String label, float[] actual, float[] expected) {
		assertThat(actual).as(label).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++)
			assertThat(actual[i]).as(label + " [%d]", i).isCloseTo(expected[i], within(TOL));
	}

	private static float[][] randomRows(Random rng, int rows, int n) {
		float[][] out = new float[rows][];
		for (int r = 0; r < rows; r++) {
			out[r] = new float[n];
			for (int i = 0; i < n; i++)
				out[r][i] = rng.nextFloat() * 2f - 1f;
		}
		return out;
	}
}
