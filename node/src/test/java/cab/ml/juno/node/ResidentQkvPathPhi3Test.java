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
 * The decode region on a Phi-3-shaped layer: Q, K and V read out of one fused
 * Q5_K tensor and gate and up out of one fused Q4_K tensor through row views, a
 * Q6_K down projection, 96-wide heads, and the extended rotation with frequency
 * factors and a magnitude scale ({@link CudaPhi3Rope}). Held against the Phi-3
 * op-at-a-time GPU path: the round-trip GPU norm, one GEMV over each fused tensor
 * then sliced on the host, {@link Phi3Rope#ropeExt} on the host, the host-packed KV
 * append and {@link CudaGqaAttention#attendBatched}, and the host's residual adds
 * and SwiGLU. Every kernel is the same on both sides and fed the same bits, so the
 * region must match bit for bit.
 *
 * <p>{@code memGetInfo} is device-wide. Run on an otherwise idle device.
 */
@Tag("gpu")
@DisplayName("Resident decode region - Phi-3 layout (fused Q/K/V and gate/up, extended RoPE)")
class ResidentQkvPathPhi3Test {

	private static final int HEADS = 8;
	private static final int KV_HEADS = 4;
	private static final int HEAD_DIM = 96;
	private static final int H = HEADS * HEAD_DIM; // 768: three Q4_K super-blocks per row
	private static final int KV = KV_HEADS * HEAD_DIM;
	private static final int INTER = 2 * H;
	private static final int LAYERS = 3;
	private static final float EPS = 1e-5f;

	private static GpuContext ctx;
	private static CudaMatVec mv;
	private static CudaRmsNorm roundTripNorm;
	private static CudaGqaAttention gqa;
	private static LlamaConfig cfg;
	private static Phi3RopeConfig rope;
	private static float[][] attnNorm;
	private static float[][] ffnNorm;
	private static DeviceQ4KMatrix[] qkv;
	private static DeviceQ4KMatrix[] wo;
	private static DeviceQ4KMatrix[] gateUp;
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

		cfg = new LlamaConfig(H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER, 1000, EPS, 10_000f, "phi3");
		Random rnd = new Random(17);
		rope = new Phi3RopeConfig(10_000f, 1.0f, 1.1902381f, 4096, 131_072, randomVec(HEAD_DIM / 2, rnd, 1f, 1.6f),
				randomVec(HEAD_DIM / 2, rnd, 1f, 40f));
		attnNorm = new float[LAYERS][];
		ffnNorm = new float[LAYERS][];
		qkv = new DeviceQ4KMatrix[LAYERS];
		wo = new DeviceQ4KMatrix[LAYERS];
		gateUp = new DeviceQ4KMatrix[LAYERS];
		down = new DeviceQ4KMatrix[LAYERS];
		for (int li = 0; li < LAYERS; li++) {
			attnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			ffnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			qkv[li] = packed(H + 2 * KV, H, QuantizationLayout.TYPE_Q5_K, rnd);
			wo[li] = packed(H, H, QuantizationLayout.TYPE_Q4_K, rnd);
			gateUp[li] = packed(2 * INTER, H, QuantizationLayout.TYPE_Q4_K, rnd);
			down[li] = packed(H, INTER, QuantizationLayout.TYPE_Q6_K, rnd);
		}
	}

	@AfterAll
	static void destroy() {
		for (DeviceQ4KMatrix[] a : new DeviceQ4KMatrix[][] { qkv, wo, gateUp, down })
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

	/** The region over the fused tensors, as the Phi-3 handler builds it. */
	private static ResidentQkvPath path() {
		DeviceQ4KMatrix[] wq = new DeviceQ4KMatrix[LAYERS];
		DeviceQ4KMatrix[] wk = new DeviceQ4KMatrix[LAYERS];
		DeviceQ4KMatrix[] wv = new DeviceQ4KMatrix[LAYERS];
		DeviceQ4KMatrix[] gate = new DeviceQ4KMatrix[LAYERS];
		DeviceQ4KMatrix[] up = new DeviceQ4KMatrix[LAYERS];
		for (int li = 0; li < LAYERS; li++) {
			wq[li] = qkv[li].rowSlice(0, H);
			wk[li] = qkv[li].rowSlice(H, KV);
			wv[li] = qkv[li].rowSlice(H + KV, KV);
			gate[li] = gateUp[li].rowSlice(0, INTER);
			up[li] = gateUp[li].rowSlice(INTER, INTER);
		}
		return ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv,
				new ResidentLayerTail.Weights(ffnNorm, wo, gate, up, down), CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, rope));
	}

	@Test
	@DisplayName("without a mirror: q, k and v match the op-at-a-time path bit for bit")
	void qkvMatchesBitForBit() {
		try (ResidentQkvPath path = path()) {
			Random rnd = new Random(31);
			for (int pos : new int[] { 0, 1, 17, 511, 4095 }) {
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
			Random rnd = new Random(32);
			for (int pos : new int[] { 0, 1, 63, 64, 517, 4095 }) {
				DeviceKvCache mine = mirrorWithHistory(pos, 800 + pos);
				DeviceKvCache ref = mirrorWithHistory(pos, 800 + pos);
				try {
					float[] x = randomVec(H, rnd, -2f, 2f);
					float[][] expected = todaysLayer(1, x, pos, ref);
					mine.ensureCapacity(pos);
					ResidentQkvPath.Output got = path.run(1, x, pos, mine);
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
			assertThat(leasedDecodeMatches(path, 30, new Random(33))).isTrue();
		}
	}

	@Test
	@DisplayName("whole layer: threads decoding in leases of their own each match")
	void concurrentLeasesEachMatch() throws Exception {
		try (ResidentQkvPath path = path()) {
			ExecutorService pool = Executors.newFixedThreadPool(3);
			List<Future<Boolean>> results = new ArrayList<>();
			for (int t = 0; t < 3; t++) {
				long seed = 400 + t;
				results.add(pool.submit(() -> leasedDecodeMatches(path, 12, new Random(seed))));
			}
			for (Future<Boolean> f : results)
				assertThat(f.get(120, TimeUnit.SECONDS)).isTrue();
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
		}
	}

	@Test
	@DisplayName("a position that needs the held-back long factors is refused, as on the CPU")
	void refusesPositionsNeedingTheLongFactors() {
		try (ResidentQkvPath path = path()) {
			assertThatThrownBy(() -> path.run(0, new float[H], 4096)).isInstanceOf(IllegalStateException.class)
					.hasMessageContaining("4096");
			// The region is usable afterwards.
			float[] x = randomVec(H, new Random(34), -2f, 2f);
			assertThat(path.run(0, x, 10)[1]).containsExactly(todaysQkv(0, x, 10)[1]);
		}
	}

	/**
	 * Decodes {@code steps} positions through every layer, one lease per token, against
	 * the op-at-a-time path on mirrors of its own. True when every layer's k, v,
	 * attention and output matched bit for bit and every layer ran whole.
	 */
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

	/** Phi-3's attention head op at a time: round-trip GPU norm, one fused GEMV sliced on the host, host rotation. */
	private static float[][] todaysQkv(int li, float[] x, int pos) {
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x }, attnNorm[li], EPS, xn)).isTrue();
		float[] fused = mv.sgemv(qkv[li], xn[0]);
		float[] q = Arrays.copyOfRange(fused, 0, H);
		float[] k = Arrays.copyOfRange(fused, H, H + KV);
		float[] v = Arrays.copyOfRange(fused, H + KV, H + 2 * KV);
		Phi3Rope.ropeExt(q, pos, HEADS, HEAD_DIM, rope);
		Phi3Rope.ropeExt(k, pos, KV_HEADS, HEAD_DIM, rope);
		return new float[][] { q, k, v };
	}

	/**
	 * Phi-3's whole decode layer op at a time: {@link #todaysQkv}, the host-packed KV
	 * append and the attention kernel, the output projection and residual add, the
	 * round-trip FFN norm, one fused gate/up GEMV sliced on the host, SwiGLU on the
	 * host, the down projection and residual add. Returns {k, v, attention, output}.
	 */
	private static float[][] todaysLayer(int li, float[] x, int pos, DeviceKvCache mirror) {
		float[][] qkvRow = todaysQkv(li, x, pos);
		mirror.appendToken(pos, qkvRow[1], qkvRow[2]);
		float[][] attn = new float[1][];
		assertThat(gqa.attendBatched(new DeviceKvCache[] { mirror }, new float[][] { qkvRow[0] },
				new int[] { pos + 1 }, attn, HEADS, HEAD_DIM, HEADS / KV_HEADS, KV)).isTrue();
		float[] x2 = LlamaTransformerHandler.add(x, mv.sgemv(wo[li], attn[0]));
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x2 }, ffnNorm[li], EPS, xn)).isTrue();
		float[] gu = mv.sgemv(gateUp[li], xn[0]);
		float[] hidden = new float[INTER];
		for (int i = 0; i < INTER; i++)
			hidden[i] = LlamaTransformerHandler.silu(gu[i]) * gu[INTER + i];
		float[] out = LlamaTransformerHandler.add(x2, mv.sgemv(down[li], hidden));
		return new float[][] { qkvRow[1], qkvRow[2], attn[0], out };
	}

	/** A mirror holding {@code tokens} random K/V rows at positions {@code [0, tokens)}. */
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
