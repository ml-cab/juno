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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Parity oracle for the real GPU-resident attention kernel
 * ({@link CudaGqaAttention#attendBatched}) against the {@link GqaMath} CPU
 * oracle — same algorithm, extracted from {@code LlamaTransformerHandler}.
 *
 * <p>The oracle reads back the <em>same</em> FP16-rounded K/V data the kernel
 * itself reads (via {@link DeviceKvCache#downloadK}/{@code downloadV}), so the
 * only expected divergence is floating-point summation-order/reduction
 * differences between the kernel's parallel reduction and the CPU's serial
 * loop — not FP16 storage rounding (already baked into both sides equally).
 * Tolerance is therefore much tighter than the cross-quantization-scheme
 * tolerances used elsewhere (e.g. {@code CudaSgemmBatchedPrefillParityTest}'s
 * {@code TOL_Q4K_BATCHED}).
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=GqaAttentionKernelParityTest}.
 */
@Tag("gpu")
@DisplayName("CudaGqaAttention.attendBatched — parity vs GqaMath CPU oracle")
class GqaAttentionKernelParityTest {

	private static final float TOL = 3e-3f;
	/** Looser tolerance at very long context: more summation terms compound rounding. */
	private static final float TOL_LONG = 8e-3f;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		assumeTrue(GqaAttentionKernel.isAvailable(), "Skipping — GPU attention kernel unavailable");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	private record Shape(String name, int numHeads, int numKvHeads, int headDim, int startPos, int window) {
		int gqaRatio() {
			return numHeads / numKvHeads;
		}

		int kvDim() {
			return numKvHeads * headDim;
		}
	}

	private static final Shape[] SHAPES = {
			new Shape("decode-first-token (GQA 8x, TinyLlama-shaped)", 32, 4, 64, 0, 1),
			new Shape("decode-mid-context (seqLen=513)", 32, 4, 64, 512, 1),
			new Shape("prefill-window (startPos=37, non-pow2 seqLens)", 32, 4, 64, 37, 5),
			new Shape("MHA-degenerate (gqaRatio=1)", 4, 4, 16, 0, 3),
			new Shape("long-context single decode (seqLen=8192)", 32, 4, 64, 8191, 1),
	};

	@Test
	@DisplayName("single-window batches match the CPU oracle within tolerance")
	void batched_prefill_or_decode_matches_cpu_oracle() {
		for (Shape shape : SHAPES) {
			runShape(shape, shape.startPos() >= 4096 ? TOL_LONG : TOL);
		}
	}

	@Test
	@DisplayName("--parallel multi-decode: independent per-stream KV caches match the CPU oracle")
	void multi_stream_independent_caches_match_cpu_oracle() {
		int numHeads = 16, numKvHeads = 2, headDim = 32;
		int gqaRatio = numHeads / numKvHeads;
		int kvDim = numKvHeads * headDim;
		int[] positions = { 10, 257, 4000 };
		int batch = positions.length;
		Random rng = new Random(99);

		DeviceKvCache[] kv = new DeviceKvCache[batch];
		float[][] qBatch = new float[batch][];
		int[] seqLens = new int[batch];
		float[][] kOracle = new float[batch][];
		float[][] vOracle = new float[batch][];
		try {
			for (int b = 0; b < batch; b++) {
				int seqLen = positions[b] + 1;
				seqLens[b] = seqLen;
				kv[b] = new DeviceKvCache(ctx, kvDim);
				float[] kFull = randomRow(rng, seqLen * kvDim);
				float[] vFull = randomRow(rng, seqLen * kvDim);
				for (int t = 0; t < seqLen; t++) {
					float[] kRow = slice(kFull, t * kvDim, kvDim);
					float[] vRow = slice(vFull, t * kvDim, kvDim);
					kv[b].appendToken(t, kRow, vRow);
				}
				qBatch[b] = randomRow(rng, numHeads * headDim);
				kOracle[b] = kv[b].downloadK(seqLen);
				vOracle[b] = kv[b].downloadV(seqLen);
			}

			CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
			float[][] gpuOut = new float[batch][];
			boolean ok = attn.attendBatched(kv, qBatch, seqLens, gpuOut, numHeads, headDim, gqaRatio, kvDim);
			assertThat(ok).as("kernel must load on a CUDA-available host").isTrue();

			for (int b = 0; b < batch; b++) {
				float[] cpuOut = new float[numHeads * headDim];
				float[] scores = new float[seqLens[b]];
				GqaMath.attend(qBatch[b], kOracle[b], vOracle[b], seqLens[b], cpuOut, scores,
						numHeads, headDim, gqaRatio, kvDim);
				assertRowsClose("stream b=" + b, gpuOut[b], cpuOut, seqLens[b] >= 4096 ? TOL_LONG : TOL);
			}
		} finally {
			for (DeviceKvCache k : kv)
				if (k != null)
					k.close();
		}
	}

	private void runShape(Shape shape, float tol) {
		Random rng = new Random(shape.name().hashCode());
		int kvDim = shape.kvDim();
		int lastPos = shape.startPos() + shape.window() - 1;

		DeviceKvCache kv = new DeviceKvCache(ctx, kvDim);
		try {
			// Write the full causal history [0, lastPos] once so every b in the
			// window has a valid, distinct seqLen view into the same cache —
			// mirrors transformerLayerBatch's growing-window dual-write.
			float[][] kRows = new float[lastPos + 1][];
			float[][] vRows = new float[lastPos + 1][];
			for (int t = 0; t <= lastPos; t++) {
				kRows[t] = randomRow(rng, kvDim);
				vRows[t] = randomRow(rng, kvDim);
				kv.appendToken(t, kRows[t], vRows[t]);
			}

			int batch = shape.window();
			float[][] qBatch = new float[batch][];
			int[] seqLens = new int[batch];
			DeviceKvCache[] kvPerB = new DeviceKvCache[batch];
			for (int b = 0; b < batch; b++) {
				qBatch[b] = randomRow(rng, shape.numHeads() * shape.headDim());
				seqLens[b] = shape.startPos() + b + 1;
				kvPerB[b] = kv; // same cache repeated — growing prefill window
			}

			CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
			float[][] gpuOut = new float[batch][];
			boolean ok = attn.attendBatched(kvPerB, qBatch, seqLens, gpuOut,
					shape.numHeads(), shape.headDim(), shape.gqaRatio(), kvDim);
			assertThat(ok).as("kernel must load on a CUDA-available host").isTrue();

			for (int b = 0; b < batch; b++) {
				int seqLen = seqLens[b];
				float[] kView = kv.downloadK(seqLen);
				float[] vView = kv.downloadV(seqLen);
				float[] cpuOut = new float[shape.numHeads() * shape.headDim()];
				float[] scores = new float[seqLen];
				GqaMath.attend(qBatch[b], kView, vView, seqLen, cpuOut, scores,
						shape.numHeads(), shape.headDim(), shape.gqaRatio(), kvDim);
				assertRowsClose(shape.name() + " b=" + b, gpuOut[b], cpuOut, tol);
			}
		} finally {
			kv.close();
		}
	}

	private static void assertRowsClose(String label, float[] actual, float[] expected, float tol) {
		assertThat(actual).as(label).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++) {
			assertThat(actual[i]).as(label + " [%d]", i).isCloseTo(expected[i], within(tol));
		}
	}

	private static float[] randomRow(Random rng, int n) {
		float[] row = new float[n];
		for (int i = 0; i < n; i++)
			row[i] = rng.nextFloat() * 2f - 1f;
		return row;
	}

	private static float[] slice(float[] src, int off, int len) {
		float[] out = new float[len];
		System.arraycopy(src, off, out, 0, len);
		return out;
	}
}
