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

import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The decode batch device region on random K-quant weights: N independent requests,
 * each at its own position over its own KV mirror, in one batched step per layer.
 * Every row must be bit for bit what that request's own single-request decode region
 * ({@link ResidentQkvPath}, itself held to the op-at-a-time path) computes, because
 * a reply must not depend on what is batched with it. Covers each shape a handler
 * runs: plain, split-half RoPE with Q/K/V biases (Qwen2), per-head Q/K norms (Qwen3)
 * and the Phi-3 family's extended RoPE.
 */
@Tag("gpu")
@DisplayName("Decode batch device region - every row equals that request's single decode")
class ResidentBatchDecodeTest {

	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 64;
	private static final int H = HEADS * HEAD_DIM;
	private static final int KV = KV_HEADS * HEAD_DIM;
	private static final int INTER = 4 * H;
	private static final int LAYERS = 3;
	private static final float THETA = 10_000f;
	private static final float EPS = 1e-5f;

	private static GpuContext ctx;
	private static LlamaConfig cfg;
	private static float[][] attnNorm;
	private static float[][] ffnNorm;
	private static DeviceQ4KMatrix[][] weights; // q, k, v, o, gate, up, down

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		assumeTrue(GqaAttentionKernel.tryLoad() != null, "attention kernel failed to load");
		cfg = new LlamaConfig(H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER, 1000, EPS, THETA, "llama");
		Random rnd = new Random(123);
		attnNorm = new float[LAYERS][];
		ffnNorm = new float[LAYERS][];
		weights = new DeviceQ4KMatrix[7][LAYERS];
		for (int li = 0; li < LAYERS; li++) {
			attnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			ffnNorm[li] = randomVec(H, rnd, 0.5f, 1.5f);
			weights[0][li] = q4k(H, H, rnd);
			weights[1][li] = q4k(KV, H, rnd);
			weights[2][li] = q4k(KV, H, rnd);
			weights[3][li] = q4k(H, H, rnd);
			weights[4][li] = q4k(INTER, H, rnd);
			weights[5][li] = q4k(INTER, H, rnd);
			weights[6][li] = q4k(H, INTER, rnd);
		}
	}

	@AfterAll
	static void destroy() {
		if (weights != null)
			for (DeviceQ4KMatrix[] w : weights)
				for (DeviceQ4KMatrix m : w)
					if (m != null)
						m.close();
		if (ctx != null)
			ctx.close();
	}

	private static ResidentQkvPath path(String shape) {
		Random rnd = new Random(55);
		ResidentLayerTail.Weights tail = new ResidentLayerTail.Weights(ffnNorm, weights[3], weights[4], weights[5],
				weights[6]);
		return switch (shape) {
		case "plain" -> ResidentQkvPath.create(ctx, cfg, attnNorm, weights[0], weights[1], weights[2], tail);
		case "qwen2" -> {
			float[][] bq = new float[LAYERS][], bk = new float[LAYERS][], bv = new float[LAYERS][];
			for (int li = 0; li < LAYERS; li++) {
				bq[li] = randomVec(H, rnd, -0.2f, 0.2f);
				bk[li] = randomVec(KV, rnd, -0.2f, 0.2f);
				bv[li] = randomVec(KV, rnd, -0.2f, 0.2f);
			}
			yield ResidentQkvPath.create(ctx, cfg, attnNorm, weights[0], weights[1], weights[2], tail,
					CudaRope.tryCreate(ctx, HEAD_DIM, THETA, RopePairing.SPLIT_HALF), null,
					new ResidentQkvPath.Biases(bq, bk, bv));
		}
		case "qwen3" -> {
			float[][] qn = new float[LAYERS][], kn = new float[LAYERS][];
			for (int li = 0; li < LAYERS; li++) {
				qn[li] = randomVec(HEAD_DIM, rnd, 0.5f, 1.5f);
				kn[li] = randomVec(HEAD_DIM, rnd, 0.5f, 1.5f);
			}
			yield ResidentQkvPath.create(ctx, cfg, attnNorm, weights[0], weights[1], weights[2], tail,
					CudaRope.tryCreate(ctx, HEAD_DIM, THETA, RopePairing.SPLIT_HALF),
					new ResidentQkvPath.HeadNorms(qn, kn));
		}
		case "phi3" -> ResidentQkvPath.create(ctx, cfg, attnNorm, weights[0], weights[1], weights[2], tail,
				CudaPhi3Rope.tryCreate(ctx, HEAD_DIM,
						new Phi3RopeConfig(10_000f, 1.0f, 1.1902381f, 4096, 131_072,
								randomVec(HEAD_DIM / 2, rnd, 1f, 1.6f), null)));
		default -> throw new IllegalArgumentException(shape);
		};
	}

	@ParameterizedTest(name = "{0}, batch of {1}")
	@CsvSource({ "plain,1", "plain,2", "plain,8", "plain,9", "qwen2,1", "qwen2,2", "qwen2,8", "qwen3,1", "qwen3,2",
			"qwen3,8", "phi3,1", "phi3,2", "phi3,8" })
	@DisplayName("k, v and the layer output of every row match its own single decode, at different positions, layer after layer")
	void everyRowMatchesItsSingleDecode(String shape, int n) {
		try (ResidentQkvPath single = path(shape); ResidentBatchDecode batch = ResidentBatchDecode.create(single)) {
			assertThat(batch).as("the batch region builds wherever the single region runs whole").isNotNull();
			Random rnd = new Random(shape.hashCode() * 31L + n);
			int[] start = new int[n];
			for (int b = 0; b < n; b++)
				start[b] = 2 + 3 * b;
			DeviceKvCache[][] mine = mirrors(n, start, rnd.nextLong());
			DeviceKvCache[][] ref = mirrors(n, start, 0);
			copyHistory(mine, ref, start);
			try {
				for (int step = 0; step < 4; step++) {
					int[] pos = new int[n];
					float[][] x = new float[n][];
					for (int b = 0; b < n; b++) {
						pos[b] = start[b] + step;
						x[b] = randomVec(H, rnd, -2f, 2f);
					}
					float[][][] expected = singleDecodes(single, x, pos, ref);
					float[][] kOut = new float[n][KV];
					float[][] vOut = new float[n][KV];
					float[][] xb = copy(x);
					try (ResidentBatchDecode.Step s = batch.open(n)) {
						assertThat(s).as("region opened").isNotNull();
						for (int li = 0; li < LAYERS; li++) {
							DeviceKvCache[] layer = layerMirrors(mine, li);
							for (DeviceKvCache m : layer)
								m.ensureCapacity(pos[0] + n + 4);
							assertThat(batch.canRun(li, pos, layer)).isTrue();
							s.runLayer(li, xb, pos, layer, kOut, vOut);
							for (int b = 0; b < n; b++) {
								String what = shape + " n=" + n + " step " + step + " layer " + li + " row " + b;
								assertBits(kOut[b], expected[li][b * 2], what + " k");
								assertBits(vOut[b], expected[li][b * 2 + 1], what + " v");
								assertBits(xb[b], expected[li + LAYERS][b], what + " layer output");
								layer[b].markWritten(pos[b], 1);
							}
						}
					}
				}
				for (int li = 0; li < LAYERS; li++)
					for (int b = 0; b < n; b++)
						assertBits(mine[b][li].downloadV(start[b] + 4), ref[b][li].downloadV(start[b] + 4),
								"V mirror request " + b + " layer " + li);
			} finally {
				closeAll(mine);
				closeAll(ref);
			}
		}
	}

	@Test
	@DisplayName("a layer the single region does not run whole is declined, and so is a mirror short of its history")
	void declinesWhatItCannotRun() {
		DeviceQ4KMatrix[] noDown = weights[6].clone();
		noDown[1] = null;
		ResidentLayerTail.Weights tail = new ResidentLayerTail.Weights(ffnNorm, weights[3], weights[4], weights[5], noDown);
		try (ResidentQkvPath single = ResidentQkvPath.create(ctx, cfg, attnNorm, weights[0], weights[1], weights[2], tail);
				ResidentBatchDecode batch = ResidentBatchDecode.create(single)) {
			DeviceKvCache m = new DeviceKvCache(ctx, KV);
			try {
				m.ensureCapacity(100);
				assertThat(batch.canRun(0, new int[] { 0 }, new DeviceKvCache[] { m })).isTrue();
				assertThat(batch.canRun(1, new int[] { 0 }, new DeviceKvCache[] { m }))
						.as("layer 1's down projection is off the device").isFalse();
				assertThat(batch.canRun(0, new int[] { 7 }, new DeviceKvCache[] { m })).as("history short of position 7")
						.isFalse();
				assertThat(batch.canRun(0, new int[] { 0 }, new DeviceKvCache[] { null })).as("no mirror").isFalse();
				m.close();
				assertThat(batch.canRun(0, new int[] { 0 }, new DeviceKvCache[] { m })).as("closed mirror").isFalse();
			} finally {
				m.close();
			}
		}
	}

	@Test
	@DisplayName("decoding allocates no device memory per step once a region is open")
	void allocatesNoDeviceMemoryPerStep() {
		try (ResidentQkvPath single = path("plain"); ResidentBatchDecode batch = ResidentBatchDecode.create(single)) {
			int n = 8;
			DeviceKvCache[][] mine = new DeviceKvCache[n][LAYERS];
			for (int b = 0; b < n; b++)
				for (int li = 0; li < LAYERS; li++) {
					mine[b][li] = new DeviceKvCache(ctx, KV);
					mine[b][li].ensureCapacity(200);
				}
			try {
				Random rnd = new Random(9);
				runStep(batch, mine, n, 0, rnd);
				long afterFirst = freeBytes();
				long bytes = batch.deviceBytes();
				for (int pos = 1; pos < 80; pos++)
					runStep(batch, mine, n, pos, rnd);
				assertThat(batch.deviceBytes()).isEqualTo(bytes).isPositive();
				assertThat(freeBytes()).isGreaterThanOrEqualTo(afterFirst);
			} finally {
				closeAll(mine);
			}
		}
	}

	private static void runStep(ResidentBatchDecode batch, DeviceKvCache[][] mirrors, int n, int pos, Random rnd) {
		int[] positions = new int[n];
		float[][] x = new float[n][];
		for (int b = 0; b < n; b++) {
			positions[b] = pos;
			x[b] = randomVec(H, rnd, -2f, 2f);
		}
		float[][] k = new float[n][KV];
		float[][] v = new float[n][KV];
		try (ResidentBatchDecode.Step s = batch.open(n)) {
			for (int li = 0; li < LAYERS; li++) {
				DeviceKvCache[] layer = layerMirrors(mirrors, li);
				s.runLayer(li, x, positions, layer, k, v);
				for (int b = 0; b < n; b++)
					layer[b].markWritten(pos, 1);
			}
		}
	}

	/**
	 * Each request decoded alone through the single-request region, in a lease per
	 * request, on its own mirrors: per layer {k, v per row}, then per layer the output
	 * rows at index {@code li + LAYERS}.
	 */
	private static float[][][] singleDecodes(ResidentQkvPath single, float[][] x, int[] pos, DeviceKvCache[][] ref) {
		int n = x.length;
		float[][][] out = new float[2 * LAYERS][][];
		for (int li = 0; li < LAYERS; li++) {
			out[li] = new float[2 * n][];
			out[li + LAYERS] = new float[n][];
		}
		for (int b = 0; b < n; b++) {
			float[] row = x[b].clone();
			try (ResidentQkvPath.Lease lease = single.lease()) {
				for (int li = 0; li < LAYERS; li++) {
					ref[b][li].ensureCapacity(pos[b]);
					ResidentQkvPath.Output got = single.run(li, row, pos[b], ref[b][li], lease);
					assertThat(got.layerDone).as("the single region runs layer %d whole", li).isTrue();
					out[li][b * 2] = got.k.clone();
					out[li][b * 2 + 1] = got.v.clone();
					out[li + LAYERS][b] = got.layer.clone();
					ref[b][li].markWritten(pos[b], 1);
					row = got.layer;
				}
			}
		}
		return out;
	}

	/** Per request, per layer: a mirror holding {@code start[b]} random rows ({@code seed} 0: left empty). */
	private static DeviceKvCache[][] mirrors(int n, int[] start, long seed) {
		DeviceKvCache[][] out = new DeviceKvCache[n][LAYERS];
		Random rnd = new Random(seed);
		for (int b = 0; b < n; b++)
			for (int li = 0; li < LAYERS; li++) {
				DeviceKvCache m = new DeviceKvCache(ctx, KV);
				m.ensureCapacity(start[b] + 16);
				if (seed != 0 && start[b] > 0) {
					float[][] k = new float[start[b]][];
					float[][] v = new float[start[b]][];
					for (int t = 0; t < start[b]; t++) {
						k[t] = randomVec(KV, rnd, -1f, 1f);
						v[t] = randomVec(KV, rnd, -1f, 1f);
					}
					m.appendWindow(0, k, v, start[b]);
				}
				out[b][li] = m;
			}
		return out;
	}

	/** Copies each history of {@code from} into {@code to}, so both start from the same rows. */
	private static void copyHistory(DeviceKvCache[][] from, DeviceKvCache[][] to, int[] start) {
		for (int b = 0; b < from.length; b++)
			for (int li = 0; li < LAYERS; li++) {
				float[] k = from[b][li].downloadK(start[b]);
				float[] v = from[b][li].downloadV(start[b]);
				float[][] kr = new float[start[b]][KV];
				float[][] vr = new float[start[b]][KV];
				for (int t = 0; t < start[b]; t++) {
					System.arraycopy(k, t * KV, kr[t], 0, KV);
					System.arraycopy(v, t * KV, vr[t], 0, KV);
				}
				to[b][li].appendWindow(0, kr, vr, start[b]);
			}
	}

	private static DeviceKvCache[] layerMirrors(DeviceKvCache[][] m, int li) {
		DeviceKvCache[] layer = new DeviceKvCache[m.length];
		for (int b = 0; b < m.length; b++)
			layer[b] = m[b][li];
		return layer;
	}

	private static void closeAll(DeviceKvCache[][] m) {
		for (DeviceKvCache[] row : m)
			for (DeviceKvCache c : row)
				if (c != null)
					c.close();
	}

	private static float[][] copy(float[][] x) {
		float[][] c = new float[x.length][];
		for (int b = 0; b < x.length; b++)
			c[b] = x[b].clone();
		return c;
	}

	private static void assertBits(float[] actual, float[] expected, String what) {
		assertThat(actual).as(what).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": element " + i + " is " + actual[i] + ", the single decode gives "
						+ expected[i]);
	}

	private static long freeBytes() {
		long[] info = ctx.bindings().memGetInfo(ctx.deviceIndex());
		assumeTrue(info[0] > 0, "device memory query unavailable");
		return info[0];
	}

	private static DeviceQ4KMatrix q4k(int rows, int cols, Random rnd) {
		float[] host = randomVec(rows * cols, rnd, -1f, 1f);
		return DeviceQ4KMatrix.upload(ctx, GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K), rows, cols);
	}

	private static float[] randomVec(int n, Random rnd, float lo, float hi) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = lo + rnd.nextFloat() * (hi - lo);
		return a;
	}
}
