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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The decode region on a Qwen3-shaped layer: an RMS norm over each head of q and
 * k between the projection and RoPE, split-half pairs, 128-wide heads, a Q6_K V and
 * down projection. Held against the Qwen3 op-at-a-time GPU path: the round-trip GPU
 * norm (over the residual row, then over each head of q and of k), the K-quant GEMVs
 * through {@link CudaMatVec}, the host rotation, the host-packed KV append and
 * {@link CudaGqaAttention#attendBatched}, and the host's residual adds and SwiGLU.
 * Every kernel is the same on both sides and fed the same bits, so the region must
 * match bit for bit.
 */
@Tag("gpu")
@DisplayName("Resident decode region - Qwen3 layout (per-head Q/K norm, split-half RoPE)")
class ResidentQkvPathQwen3Test {

	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 128;
	private static final int H = HEADS * HEAD_DIM; // 512: two Q4_K super-blocks per row
	private static final int KV = KV_HEADS * HEAD_DIM;
	private static final int INTER = 3 * H;
	private static final int LAYERS = 3;
	private static final float EPS = 1e-6f;
	private static final float THETA = 1_000_000f;

	private static GpuContext ctx;
	private static CudaMatVec mv;
	private static CudaRmsNorm roundTripNorm;
	private static CudaGqaAttention gqa;
	private static LlamaConfig cfg;
	private static float[][] attnNorm;
	private static float[][] ffnNorm;
	private static float[][] qNorm;
	private static float[][] kNorm;
	private static DeviceQ4KMatrix[] wq;
	private static DeviceQ4KMatrix[] wk;
	private static DeviceQ4KMatrix[] wv;
	private static DeviceQ4KMatrix[] wo;
	private static DeviceQ4KMatrix[] gate;
	private static DeviceQ4KMatrix[] up;
	private static DeviceQ4KMatrix[] down;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "K-quant MMQ kernel failed to load");
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
		assumeTrue(GqaAttentionKernel.tryLoad() != null, "attention kernel failed to load");
		mv = new CudaMatVec(ctx);
		roundTripNorm = CudaRmsNorm.tryCreate(ctx);
		gqa = CudaGqaAttention.tryCreate(ctx);

		cfg = new LlamaConfig(H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER, 1000, EPS, THETA, "qwen3");
		Random rnd = new Random(23);
		attnNorm = new float[LAYERS][];
		ffnNorm = new float[LAYERS][];
		qNorm = new float[LAYERS][];
		kNorm = new float[LAYERS][];
		wq = new DeviceQ4KMatrix[LAYERS];
		wk = new DeviceQ4KMatrix[LAYERS];
		wv = new DeviceQ4KMatrix[LAYERS];
		wo = new DeviceQ4KMatrix[LAYERS];
		gate = new DeviceQ4KMatrix[LAYERS];
		up = new DeviceQ4KMatrix[LAYERS];
		down = new DeviceQ4KMatrix[LAYERS];
		for (int li = 0; li < LAYERS; li++) {
			attnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			ffnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			qNorm[li] = randomVec(HEAD_DIM, rnd, 0.5f, 2.5f);
			kNorm[li] = randomVec(HEAD_DIM, rnd, 0.5f, 2.5f);
			wq[li] = packed(H, H, QuantizationLayout.TYPE_Q4_K, rnd);
			wk[li] = packed(KV, H, QuantizationLayout.TYPE_Q4_K, rnd);
			wv[li] = packed(KV, H, QuantizationLayout.TYPE_Q6_K, rnd);
			wo[li] = packed(H, H, QuantizationLayout.TYPE_Q4_K, rnd);
			gate[li] = packed(INTER, H, QuantizationLayout.TYPE_Q4_K, rnd);
			up[li] = packed(INTER, H, QuantizationLayout.TYPE_Q4_K, rnd);
			down[li] = packed(H, INTER, QuantizationLayout.TYPE_Q6_K, rnd);
		}
	}

	@AfterAll
	static void destroy() {
		for (DeviceQ4KMatrix[] a : new DeviceQ4KMatrix[][] { wq, wk, wv, wo, gate, up, down })
			if (a != null)
				for (DeviceQ4KMatrix m : a)
					if (m != null)
						m.close();
		if (gqa != null)
			gqa.close();
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	/** The region as the Qwen3 handler builds it. */
	private static ResidentQkvPath path() {
		return ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv,
				new ResidentLayerTail.Weights(ffnNorm, wo, gate, up, down),
				CudaRope.tryCreate(ctx, HEAD_DIM, THETA, RopePairing.SPLIT_HALF),
				new ResidentQkvPath.HeadNorms(qNorm, kNorm));
	}

	@Test
	@DisplayName("without a mirror: q, k and v match the op-at-a-time path bit for bit")
	void qkvMatchesBitForBit() {
		try (ResidentQkvPath path = path()) {
			Random rnd = new Random(41);
			for (int pos : new int[] { 0, 1, 17, 511, 30_000 }) {
				float[] x = randomVec(H, rnd, -2f, 2f);
				float[][] expected = todaysQkv(0, x, pos);
				float[][] got = path.run(0, x, pos);
				assertThat(got[0]).as("q at pos %d", pos).containsExactly(expected[0]);
				assertThat(got[1]).as("k at pos %d", pos).containsExactly(expected[1]);
				assertThat(got[2]).as("v at pos %d", pos).containsExactly(expected[2]);
			}
		}
	}

	@Test
	@DisplayName("whole layer: k, v, attention and the layer output match the op-at-a-time path bit for bit")
	void wholeLayerMatchesBitForBit() {
		try (ResidentQkvPath path = path()) {
			for (int li = 0; li < LAYERS; li++)
				assertThat(path.runsWholeLayer(li)).as("layer %d", li).isTrue();
			Random rnd = new Random(42);
			for (int pos : new int[] { 0, 1, 63, 64, 517, 4096 }) {
				DeviceKvCache mine = mirrorWithHistory(pos, 600 + pos);
				DeviceKvCache ref = mirrorWithHistory(pos, 600 + pos);
				try {
					float[] x = randomVec(H, rnd, -2f, 2f);
					float[][] expected = todaysLayer(2, x, pos, ref);
					mine.ensureCapacity(pos);
					ResidentQkvPath.Output got = path.run(2, x, pos, mine);
					assertThat(got.layerDone).as("whole layer at pos %d", pos).isTrue();
					assertThat(got.k).as("k at pos %d", pos).containsExactly(expected[0]);
					assertThat(got.v).as("v at pos %d", pos).containsExactly(expected[1]);
					assertThat(got.attn).as("attention at pos %d", pos).containsExactly(expected[2]);
					assertThat(got.layer).as("layer output at pos %d", pos).containsExactly(expected[3]);
				} finally {
					mine.close();
					ref.close();
				}
			}
		}
	}

	@Test
	@DisplayName("whole layer: a 30-token decode through three layers in a lease matches layer for layer")
	void leasedDecodeMatches() {
		try (ResidentQkvPath path = path()) {
			assertThat(leasedDecodeMatches(path, 30, new Random(43))).isTrue();
		}
	}

	@Test
	@DisplayName("whole layer: threads decoding in leases of their own each match")
	void concurrentLeasesEachMatch() throws Exception {
		try (ResidentQkvPath path = path()) {
			ExecutorService pool = Executors.newFixedThreadPool(3);
			List<Future<Boolean>> results = new ArrayList<>();
			for (int t = 0; t < 3; t++) {
				long seed = 500 + t;
				results.add(pool.submit(() -> leasedDecodeMatches(path, 12, new Random(seed))));
			}
			for (Future<Boolean> f : results)
				assertThat(f.get(120, TimeUnit.SECONDS)).isTrue();
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
		}
	}

	@Test
	@DisplayName("a layer without both head norms is not eligible: it is declined, not run without them")
	void aLayerWithoutHeadNormsIsDeclined() {
		float[][] partialK = { kNorm[0], null, kNorm[2] };
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv,
				new ResidentLayerTail.Weights(ffnNorm, wo, gate, up, down),
				CudaRope.tryCreate(ctx, HEAD_DIM, THETA, RopePairing.SPLIT_HALF),
				new ResidentQkvPath.HeadNorms(qNorm, partialK))) {
			assertThat(path.eligible(0)).isTrue();
			assertThat(path.eligible(1)).isFalse();
			assertThat(path.run(1, new float[H], 0)).isNull();
		}
	}

	private static boolean leasedDecodeMatches(ResidentQkvPath path, int steps, Random rnd) {
		DeviceKvCache[] mine = new DeviceKvCache[LAYERS];
		DeviceKvCache[] ref = new DeviceKvCache[LAYERS];
		for (int li = 0; li < LAYERS; li++) {
			mine[li] = new DeviceKvCache(ctx, KV);
			ref[li] = new DeviceKvCache(ctx, KV);
		}
		try {
			for (int pos = 0; pos < steps; pos++) {
				float[] x = randomVec(H, rnd, -2f, 2f);
				try (ResidentQkvPath.Lease lease = path.lease()) {
					for (int li = 0; li < LAYERS; li++) {
						float[][] expected = todaysLayer(li, x, pos, ref[li]);
						mine[li].ensureCapacity(pos);
						ResidentQkvPath.Output got = path.run(li, x, pos, mine[li], lease);
						if (!got.layerDone || !Arrays.equals(got.k, expected[0]) || !Arrays.equals(got.v, expected[1])
								|| !Arrays.equals(got.attn, expected[2]) || !Arrays.equals(got.layer, expected[3]))
							return false;
						mine[li].markWritten(pos, 1);
						x = got.layer;
					}
				}
			}
			for (int li = 0; li < LAYERS; li++)
				if (!Arrays.equals(mine[li].downloadK(steps), ref[li].downloadK(steps))
						|| !Arrays.equals(mine[li].downloadV(steps), ref[li].downloadV(steps)))
					return false;
			return true;
		} finally {
			for (int li = 0; li < LAYERS; li++) {
				mine[li].close();
				ref[li].close();
			}
		}
	}

	/**
	 * Qwen3's attention head op at a time on the GPU: round-trip norm, three K-quant
	 * GEMVs, the round-trip norm over each head of q and of k, the host rotation.
	 */
	private static float[][] todaysQkv(int li, float[] x, int pos) {
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x }, attnNorm[li], EPS, xn)).isTrue();
		float[] q = perHead(mv.sgemv(wq[li], xn[0]), qNorm[li], HEADS);
		float[] k = perHead(mv.sgemv(wk[li], xn[0]), kNorm[li], KV_HEADS);
		float[] v = mv.sgemv(wv[li], xn[0]);
		LlamaTransformerHandler.rope(q, pos, HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);
		LlamaTransformerHandler.rope(k, pos, KV_HEADS, HEAD_DIM, THETA, RopePairing.SPLIT_HALF);
		return new float[][] { q, k, v };
	}

	/** The GPU norm kernel over each {@code HEAD_DIM}-wide head of {@code row}, one round trip. */
	private static float[] perHead(float[] row, float[] weight, int heads) {
		float[][] in = new float[heads][];
		for (int h = 0; h < heads; h++)
			in[h] = Arrays.copyOfRange(row, h * HEAD_DIM, (h + 1) * HEAD_DIM);
		float[][] out = new float[heads][];
		assertThat(roundTripNorm.normalizeBatch(in, weight, EPS, out)).isTrue();
		float[] joined = new float[heads * HEAD_DIM];
		for (int h = 0; h < heads; h++)
			System.arraycopy(out[h], 0, joined, h * HEAD_DIM, HEAD_DIM);
		return joined;
	}

	/** Qwen3's whole decode layer op at a time on the GPU. Returns {k, v, attention, output}. */
	private static float[][] todaysLayer(int li, float[] x, int pos, DeviceKvCache mirror) {
		float[][] qkvRow = todaysQkv(li, x, pos);
		mirror.appendToken(pos, qkvRow[1], qkvRow[2]);
		float[][] attn = new float[1][];
		assertThat(gqa.attendBatched(new DeviceKvCache[] { mirror }, new float[][] { qkvRow[0] },
				new int[] { pos + 1 }, attn, HEADS, HEAD_DIM, HEADS / KV_HEADS, KV)).isTrue();
		float[] x2 = LlamaTransformerHandler.add(x, mv.sgemv(wo[li], attn[0]));
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x2 }, ffnNorm[li], EPS, xn)).isTrue();
		float[] g = mv.sgemv(gate[li], xn[0]);
		float[] u = mv.sgemv(up[li], xn[0]);
		float[] hidden = new float[INTER];
		for (int i = 0; i < INTER; i++)
			hidden[i] = LlamaTransformerHandler.silu(g[i]) * u[i];
		float[] out = LlamaTransformerHandler.add(x2, mv.sgemv(down[li], hidden));
		return new float[][] { qkvRow[1], qkvRow[2], attn[0], out };
	}

	private static DeviceKvCache mirrorWithHistory(int tokens, long seed) {
		DeviceKvCache m = new DeviceKvCache(ctx, KV);
		if (tokens > 0) {
			Random rnd = new Random(seed);
			float[][] k = new float[tokens][];
			float[][] v = new float[tokens][];
			for (int t = 0; t < tokens; t++) {
				k[t] = randomVec(KV, rnd, -1f, 1f);
				v[t] = randomVec(KV, rnd, -1f, 1f);
			}
			m.appendWindow(0, k, v, tokens);
		}
		return m;
	}

	private static DeviceQ4KMatrix packed(int rows, int cols, int type, Random rnd) {
		byte[] raw = GgufKQuantCodec.encode(randomVec(rows * cols, rnd, -1f, 1f), type);
		return DeviceQ4KMatrix.upload(ctx, raw, rows, cols, type);
	}

	private static float[] randomVec(int n, Random rnd, float lo, float hi) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = lo + rnd.nextFloat() * (hi - lo);
		return a;
	}
}
