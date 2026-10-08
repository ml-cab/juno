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
 * The GPU attention kernel at long context, against the scalar oracle
 * ({@link GqaMath#attend}, itself held to an independent reference by
 * {@code GqaMathOracleTest}).
 *
 * <p>Covers what {@code GqaAttentionKernelParityTest} does not: prefill windows
 * as wide as a 2048-token prompt, a single decode row at
 * {@link DeviceKvCache#MAX_SEQ_LEN}, windows that start and end off a key-tile
 * boundary, every head width a supported handler uses (64, 80, 96, 128) and a
 * 256-wide one, and the per-layer attention window.
 *
 * <p>It also holds the kernel's device scratch to the property a streaming
 * (online-softmax) kernel exists to buy: at a fixed number of query rows, the
 * scratch beyond the query and output rows does not grow with the context (so
 * doubling the sequence length less than doubles the whole), and at the longest
 * tested length it is at least 80% below what
 * a kernel that materializes every score needs ({@code rows x heads x seqLen}
 * floats on top of the query and output rows).
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=GqaAttentionTiledTest}.
 */
@Tag("gpu")
@DisplayName("GPU attention kernel: long context, head widths, window, scratch")
class GqaAttentionTiledTest {

	private static final float TOL = 3e-3f;
	/** More summed keys compound float rounding on both sides. */
	private static final float TOL_LONG = 8e-3f;

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
	private static final Shape MISTRAL = new Shape("Mistral 7B 32/8/128", 32, 8, 128);
	private static final Shape QWEN2 = new Shape("Qwen2.5-3B 16/2/128", 16, 2, 128);
	private static final Shape PHI3 = new Shape("Phi-3.5-mini 32/32/96", 32, 32, 96);
	private static final Shape PHI2 = new Shape("Phi-2 32/32/80", 32, 32, 80);
	private static final Shape WIDE = new Shape("256-wide heads 8/2/256", 8, 2, 256);

	@Test
	@DisplayName("a 2048-row prefill window from position 0 matches the oracle (TinyLlama and Mistral 7B shapes)")
	void wholePromptWindowMatchesOracle() {
		runWindow(TINYLLAMA, 0, 2048, 0, TOL_LONG);
		runWindow(MISTRAL, 0, 2048, 0, TOL_LONG);
	}

	@Test
	@DisplayName("windows off the key-tile boundary match the oracle at every head width")
	void unalignedWindowsMatchOracle() {
		runWindow(TINYLLAMA, 0, 1, 0, TOL);
		runWindow(TINYLLAMA, 31, 33, 0, TOL);
		runWindow(MISTRAL, 63, 65, 0, TOL);
		runWindow(PHI3, 100, 600, 0, TOL);
		runWindow(PHI2, 7, 300, 0, TOL);
		runWindow(WIDE, 0, 100, 0, TOL);
		runWindow(WIDE, 1000, 37, 0, TOL);
	}

	@Test
	@DisplayName("a decode row at MAX_SEQ_LEN matches the oracle")
	void decodeAtMaxSeqLenMatchesOracle() {
		runWindow(TINYLLAMA, DeviceKvCache.MAX_SEQ_LEN - 1, 1, 0, TOL_LONG);
		runWindow(MISTRAL, DeviceKvCache.MAX_SEQ_LEN - 1, 1, 0, TOL_LONG);
	}

	@Test
	@DisplayName("a decode row matches the oracle at every head width and short to long context")
	void decodeRowMatchesOracleAtEveryHeadWidth() {
		for (Shape shape : new Shape[] { TINYLLAMA, QWEN2, MISTRAL, PHI3, PHI2, WIDE })
			for (int seqLen : new int[] { 1, 31, 33, 576, 2049 })
				runWindow(shape, seqLen - 1, 1, 0, seqLen > 1024 ? TOL_LONG : TOL);
	}

	@Test
	@DisplayName("a 64-row window deep into a long prompt matches the oracle")
	void deepWindowMatchesOracle() {
		runWindow(MISTRAL, 8192 - 64, 64, 0, TOL_LONG);
	}

	@Test
	@DisplayName("independent --parallel streams of very different lengths match the oracle")
	void streamsOfMixedLengthMatchOracle() {
		Shape shape = MISTRAL;
		int[] seqLens = { 1, 2, 33, 700, 4097 };
		int batch = seqLens.length;
		Random rng = new Random(7);
		DeviceKvCache[] kv = new DeviceKvCache[batch];
		float[][] q = new float[batch][];
		float[][] kHost = new float[batch][];
		float[][] vHost = new float[batch][];
		try {
			for (int b = 0; b < batch; b++) {
				kv[b] = new DeviceKvCache(ctx, shape.kvDim());
				fill(kv[b], rng, seqLens[b], shape.kvDim());
				kHost[b] = kv[b].downloadK(seqLens[b]);
				vHost[b] = kv[b].downloadV(seqLens[b]);
				q[b] = randomRow(rng, shape.rowDim());
			}
			float[][] out = attend(shape, kv, q, seqLens, 0);
			for (int b = 0; b < batch; b++)
				assertClose("stream " + b, out[b], oracle(shape, q[b], kHost[b], vHost[b], 0, seqLens[b]),
						seqLens[b] >= 4096 ? TOL_LONG : TOL);
		} finally {
			for (DeviceKvCache k : kv)
				if (k != null)
					k.close();
		}
	}

	@Test
	@DisplayName("window 0 (none) is bit-identical to a window no shorter than any row's context")
	void noWindowIsBitIdenticalToAnUnboundingWindow() {
		Shape shape = MISTRAL;
		Random rng = new Random(11);
		int start = 200, rows = 77;
		try (DeviceKvCache kv = new DeviceKvCache(ctx, shape.kvDim())) {
			fill(kv, rng, start + rows, shape.kvDim());
			DeviceKvCache[] perRow = repeat(kv, rows);
			float[][] q = randomRows(rng, rows, shape.rowDim());
			int[] seqLens = causal(start, rows);
			float[][] none = attend(shape, perRow, q, seqLens, 0);
			float[][] wide = attend(shape, perRow, q, seqLens, start + rows);
			float[][] max = attend(shape, perRow, q, seqLens, DeviceKvCache.MAX_SEQ_LEN);
			for (int b = 0; b < rows; b++) {
				assertThat(wide[b]).as("row " + b + ", window = context").containsExactly(none[b]);
				assertThat(max[b]).as("row " + b + ", window = MAX_SEQ_LEN").containsExactly(none[b]);
			}
		}
	}

	@Test
	@DisplayName("a window shorter than the context attends over exactly its last rows")
	void windowAttendsOverTheLastRows() {
		for (int window : new int[] { 1, 31, 32, 33, 500 }) {
			runWindow(MISTRAL, 600, 70, window, TOL);
			runWindow(TINYLLAMA, 0, 100, window, TOL);
		}
	}

	@Test
	@DisplayName("device scratch: doubling the context less than doubles it; >= 80% below a materializing kernel")
	void scratchDoesNotGrowWithContext() {
		Shape shape = TINYLLAMA;
		int rows = 64;
		long rowPart = materializingScratch(shape, rows, 0);
		long previous = -1;
		int previousLen = 0;
		for (int seqLen = 1024; seqLen <= DeviceKvCache.MAX_SEQ_LEN; seqLen *= 2) {
			long scratch = scratchAfterOneCall(shape, seqLen - rows, rows);
			long materializing = materializingScratch(shape, rows, seqLen);
			System.out.printf("attention scratch: %d rows at seqLen %d: %d bytes (materializing kernel: %d)%n",
					rows, seqLen, scratch, materializing);
			// The part beyond the query, output and table rows must not grow with the
			// context at all: stricter than "doubling less than doubles it", which the
			// fixed row part alone would let a materializing kernel pass.
			if (previous >= 0)
				assertThat(scratch - rowPart)
						.as("context-dependent scratch at seqLen %d against %d at %d", seqLen, previous - rowPart,
								previousLen)
						.isLessThanOrEqualTo(previous - rowPart);
			if (seqLen == DeviceKvCache.MAX_SEQ_LEN)
				assertThat((double) scratch).as("scratch at MAX_SEQ_LEN")
						.isLessThanOrEqualTo(0.20 * materializing);
			previous = scratch;
			previousLen = seqLen;
		}
		long whole = scratchAfterOneCall(shape, 0, 2048);
		System.out.printf("attention scratch: 2048 rows at seqLen 2048: %d bytes (materializing kernel: %d)%n",
				whole, materializingScratch(shape, 2048, 2048));
		assertThat((double) whole).isLessThanOrEqualTo(0.20 * materializingScratch(shape, 2048, 2048));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	/**
	 * Scratch a kernel that keeps every score holds for one call: the query and
	 * output rows, the scores, and the pointer and length tables.
	 */
	private static long materializingScratch(Shape shape, int rows, int seqLen) {
		long qOut = 2L * rows * shape.rowDim() * Float.BYTES;
		long scores = (long) rows * shape.numHeads() * seqLen * Float.BYTES;
		long tables = (long) rows * (2 * 8 + Integer.BYTES);
		return qOut + scores + tables;
	}

	private long scratchAfterOneCall(Shape shape, int startPos, int rows) {
		Random rng = new Random(startPos * 31L + rows);
		try (DeviceKvCache kv = new DeviceKvCache(ctx, shape.kvDim())) {
			fill(kv, rng, startPos + rows, shape.kvDim());
			CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
			try {
				float[][] out = new float[rows][];
				boolean ok = attn.attendBatched(repeat(kv, rows), randomRows(rng, rows, shape.rowDim()),
						causal(startPos, rows), out, shape.numHeads(), shape.headDim(), shape.gqaRatio(),
						shape.kvDim(), 0);
				assertThat(ok).isTrue();
				return attn.scratchDeviceBytes();
			} finally {
				attn.close();
			}
		}
	}

	/**
	 * One growing prefill window of {@code rows} rows at positions
	 * {@code startPos ...} over one cache, every row (or, for wide windows, a
	 * spread of rows including both ends) checked against the oracle.
	 */
	private void runWindow(Shape shape, int startPos, int rows, int window, float tol) {
		Random rng = new Random(shape.name().hashCode() * 31L + startPos * 7L + rows + window);
		int total = startPos + rows;
		try (DeviceKvCache kv = new DeviceKvCache(ctx, shape.kvDim())) {
			fill(kv, rng, total, shape.kvDim());
			float[] kHost = kv.downloadK(total);
			float[] vHost = kv.downloadV(total);
			float[][] q = randomRows(rng, rows, shape.rowDim());
			int[] seqLens = causal(startPos, rows);
			float[][] out = attend(shape, repeat(kv, rows), q, seqLens, window);
			int stride = rows <= 128 ? 1 : rows / 41;
			for (int b = 0; b < rows; b++) {
				if (b % stride != 0 && b < rows - 3)
					continue;
				int from = window > 0 && seqLens[b] > window ? seqLens[b] - window : 0;
				assertClose(shape.name() + " start " + startPos + " window " + window + " row " + b, out[b],
						oracle(shape, q[b], kHost, vHost, from, seqLens[b]), tol);
			}
		}
	}

	private static float[][] attend(Shape shape, DeviceKvCache[] kv, float[][] q, int[] seqLens, int window) {
		CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
		try {
			float[][] out = new float[q.length][];
			boolean ok = attn.attendBatched(kv, q, seqLens, out, shape.numHeads(), shape.headDim(),
					shape.gqaRatio(), shape.kvDim(), window);
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

	private static void fill(DeviceKvCache kv, Random rng, int count, int kvDim) {
		int chunk = 4096;
		for (int start = 0; start < count; start += chunk) {
			int n = Math.min(chunk, count - start);
			kv.appendWindow(start, randomRows(rng, n, kvDim), randomRows(rng, n, kvDim), n);
		}
	}

	private static DeviceKvCache[] repeat(DeviceKvCache kv, int n) {
		DeviceKvCache[] out = new DeviceKvCache[n];
		java.util.Arrays.fill(out, kv);
		return out;
	}

	private static int[] causal(int startPos, int rows) {
		int[] seqLens = new int[rows];
		for (int b = 0; b < rows; b++)
			seqLens[b] = startPos + b + 1;
		return seqLens;
	}

	private static void assertClose(String label, float[] actual, float[] expected, float tol) {
		assertThat(actual).as(label).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++)
			assertThat(actual[i]).as(label + " [%d]", i).isCloseTo(expected[i], within(tol));
	}

	private static float[][] randomRows(Random rng, int rows, int n) {
		float[][] out = new float[rows][];
		for (int r = 0; r < rows; r++)
			out[r] = randomRow(rng, n);
		return out;
	}

	private static float[] randomRow(Random rng, int n) {
		float[] row = new float[n];
		for (int i = 0; i < n; i++)
			row[i] = rng.nextFloat() * 2f - 1f;
		return row;
	}
}
