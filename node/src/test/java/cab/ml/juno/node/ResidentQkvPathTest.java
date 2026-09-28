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
 * The decode region norm -> Q/K/V projection -> RoPE on one device-resident
 * activation, one upload in and one download of q, k and v out, against the path
 * the handler runs today: GPU RMS norm with its own round trip, the K-quant
 * projection through {@link CudaMatVec#sgemvSameX(DeviceQ4KMatrix[], float[])},
 * and the scalar CPU rotation. Every kernel is the same on both sides and fed
 * the same bits, so the region must match bit for bit.
 *
 * <p>{@code memGetInfo} is device-wide. Run on an otherwise idle device.
 */
@Tag("gpu")
@DisplayName("Resident norm -> QKV -> RoPE decode region")
class ResidentQkvPathTest {

	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 64;
	private static final int H = HEADS * HEAD_DIM; // 256: one Q4_K super-block per row
	private static final int KV = KV_HEADS * HEAD_DIM;
	private static final float THETA = 10_000f;
	private static final float EPS = 1e-5f;

	private static GpuContext ctx;
	private static CudaMatVec mv;
	private static CudaRmsNorm roundTripNorm;
	private static LlamaConfig cfg;
	private static float[][] attnNorm;
	private static DeviceQ4KMatrix[] wq;
	private static DeviceQ4KMatrix[] wk;
	private static DeviceQ4KMatrix[] wv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
		mv = new CudaMatVec(ctx);
		roundTripNorm = CudaRmsNorm.tryCreate(ctx);
		assumeTrue(roundTripNorm != null, "CUDA backend required");

		cfg = new LlamaConfig(H, 2, HEADS, KV_HEADS, HEAD_DIM, 4 * H, 1000, EPS, THETA, "llama");
		Random rnd = new Random(7);
		attnNorm = new float[][] { randomVec(H, rnd, 0.5f, 1.5f), randomVec(H, rnd, 0.5f, 1.5f) };
		// Layer 1 has no K-quant Q projection on the device: the region must decline it.
		wq = new DeviceQ4KMatrix[] { q4k(H, H, rnd), null };
		wk = new DeviceQ4KMatrix[] { q4k(KV, H, rnd), q4k(KV, H, rnd) };
		wv = new DeviceQ4KMatrix[] { q4k(KV, H, rnd), q4k(KV, H, rnd) };
	}

	@AfterAll
	static void destroy() {
		for (DeviceQ4KMatrix[] a : new DeviceQ4KMatrix[][] { wq, wk, wv })
			if (a != null)
				for (DeviceQ4KMatrix m : a)
					if (m != null)
						m.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("q, k and v match the op-at-a-time path bit for bit, at several positions")
	void matchesTodaysPathBitForBit() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			Random rnd = new Random(11);
			for (int pos : new int[] { 0, 1, 17, 511, 30_000 }) {
				float[] x = randomVec(H, rnd, -2f, 2f);
				float[][] expected = todaysPath(x, 0, pos);
				float[][] got = path.run(0, x, pos);
				assertThat(got).as("layer 0 is eligible").isNotNull();
				assertThat(got[0]).as("q at pos %d", pos).containsExactly(expected[0]);
				assertThat(got[1]).as("k at pos %d", pos).containsExactly(expected[1]);
				assertThat(got[2]).as("v at pos %d", pos).containsExactly(expected[2]);
			}
		}
	}

	@Test
	@DisplayName("the input activation is never written through")
	void leavesTheInputAlone() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			float[] x = randomVec(H, new Random(3), -2f, 2f);
			float[] copy = x.clone();
			path.run(0, x, 5);
			assertThat(x).containsExactly(copy);
		}
	}

	@Test
	@DisplayName("a layer without K-quant device projections is declined, not half-run")
	void declinesALayerItCannotRun() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			assertThat(path.eligible(0)).isTrue();
			assertThat(path.eligible(1)).isFalse();
			assertThat(path.run(1, new float[H], 0)).isNull();
		}
	}

	@Test
	@DisplayName("threads get regions of their own and still match")
	void concurrentCallersEachMatch() throws Exception {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			ExecutorService pool = Executors.newFixedThreadPool(3);
			List<Future<Boolean>> results = new ArrayList<>();
			for (int t = 0; t < 3; t++) {
				long seed = 100 + t;
				results.add(pool.submit(() -> {
					Random rnd = new Random(seed);
					for (int k = 0; k < 20; k++) {
						int pos = rnd.nextInt(4096);
						float[] x = randomVec(H, rnd, -2f, 2f);
						float[][] expected = todaysPath(x, 0, pos);
						float[][] got = path.run(0, x, pos);
						if (!java.util.Arrays.equals(got[0], expected[0]) || !java.util.Arrays.equals(got[1], expected[1])
								|| !java.util.Arrays.equals(got[2], expected[2]))
							return false;
					}
					return true;
				}));
			}
			for (Future<Boolean> f : results)
				assertThat(f.get(60, TimeUnit.SECONDS)).isTrue();
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
		}
	}

	@Test
	@DisplayName("repeated calls reuse one region: no device memory is allocated per call")
	void noDeviceAllocationPerCall() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			float[] x = randomVec(H, new Random(5), -2f, 2f);
			path.run(0, x, 0);
			long afterFirst = freeBytes();
			long bytes = path.deviceBytes();
			for (int i = 1; i < 200; i++)
				path.run(0, x, i);
			assertThat(path.deviceBytes()).isEqualTo(bytes).isPositive();
			assertThat(freeBytes()).isEqualTo(afterFirst);
		}
	}

	@Test
	@DisplayName("threads that come and go do not each keep a region: device regions are pooled")
	void shortLivedThreadsDoNotAccumulateRegions() throws Exception {
		// A server may run every request on a new thread. Device memory tied to a
		// thread that has ended is never used again, so it must not be per thread.
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			float[] x = randomVec(H, new Random(8), -2f, 2f);
			path.run(0, x, 0);
			long oneRegion = path.deviceBytes();
			for (int t = 0; t < 50; t++) {
				Thread th = new Thread(() -> path.run(0, x, 1));
				th.start();
				th.join();
			}
			assertThat(path.deviceBytes()).as("50 sequential short-lived threads reuse one region")
					.isEqualTo(oneRegion);
		}
	}

	@Test
	@DisplayName("device memory returns to where it started across 300 create-run-close cycles")
	void noDeviceMemoryRetained() {
		// One region is a few kilobytes (a stream, six small buffers, the norm
		// weight), below what one device-memory reading resolves, so one cycle
		// cannot show a leak. Measured on the reference host: with every region
		// deliberately leaked, 300 cycles hold 48 MB; without a leak the reading
		// moves by at most 256 KB either way as the driver's pool settles, and not
		// with the cycle count. The bound sits between the two.
		float[] x = randomVec(H, new Random(6), -2f, 2f);
		for (int w = 0; w < 5; w++)
			try (ResidentQkvPath warm = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
				warm.run(0, x, 0);
			}
		long before = freeBytes();
		for (int c = 0; c < 300; c++) {
			try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
				path.run(0, x, c);
			}
		}
		assertThat(Math.abs(before - freeBytes())).as("device bytes not returned after 300 cycles")
				.isLessThanOrEqualTo(4L * 1024 * 1024);
	}

	@Test
	@DisplayName("architectures and layouts the device path cannot run are named, not silently accepted")
	void namesWhatItCannotRun() {
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.ADJACENT, false)).isNull();
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.SPLIT_HALF, false)).contains("split-half");
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.ADJACENT, true)).contains("bias");
		assertThat(ResidentQkvPath.unsupportedReason(null, RopePairing.ADJACENT, false)).contains("CUDA");
	}

	/** Today's single-decode path: round-trip GPU norm, sgemvSameX over K-quant weights, CPU rotation. */
	private static float[][] todaysPath(float[] x, int li, int pos) {
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x }, attnNorm[li], EPS, xn)).isTrue();
		float[][] qkv = mv.sgemvSameX(new DeviceQ4KMatrix[] { wq[li], wk[li], wv[li] }, xn[0]);
		LlamaTransformerHandler.rope(qkv[0], pos, HEADS, HEAD_DIM, THETA);
		LlamaTransformerHandler.rope(qkv[1], pos, KV_HEADS, HEAD_DIM, THETA);
		return qkv;
	}

	private static DeviceQ4KMatrix q4k(int rows, int cols, Random rnd) {
		float[] host = randomVec(rows * cols, rnd, -1f, 1f);
		byte[] raw = GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K);
		return DeviceQ4KMatrix.upload(ctx, raw, rows, cols);
	}

	private static float[] randomVec(int n, Random rnd, float lo, float hi) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = lo + rnd.nextFloat() * (hi - lo);
		return a;
	}

	private static long freeBytes() {
		long[] info = ctx.bindings().memGetInfo(ctx.deviceIndex());
		assumeTrue(info[0] > 0, "device memory query unavailable");
		return info[0];
	}
}
