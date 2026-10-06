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
	private static CudaGqaAttention gqa;
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
		assumeTrue(GqaAttentionKernel.tryLoad() != null, "attention kernel failed to load");
		gqa = CudaGqaAttention.tryCreate(ctx);

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
		if (wholeLoaded)
			for (DeviceQ4KMatrix[] a : new DeviceQ4KMatrix[][] { Whole.WQ, Whole.WK, Whole.WV, Whole.WO, Whole.GATE,
					Whole.UP, Whole.DOWN })
				for (DeviceQ4KMatrix m : a)
					if (m != null)
						m.close();
		if (offRegionDown != null)
			offRegionDown.close();
		if (gqa != null)
			gqa.close();
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
		// One region is a few kilobytes (a stream, a few small buffers, the norm
		// weight), below what one device-memory reading resolves, so one cycle
		// cannot show a leak. Measured on the reference host: with every region
		// deliberately leaked, 300 cycles hold 48 MB; without a leak the reading
		// moves by at most 256 KB either way as the driver's pool settles, and not
		// with the cycle count. The bound sits between the two.
		float[] x = randomVec(H, new Random(6), -2f, 2f);
		for (int w = 0; w < 5; w++)
			try (ResidentQkvPath warm = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv);
					DeviceKvCache m = new DeviceKvCache(ctx, KV)) {
				warm.run(0, x, 0);
				warm.run(0, x, 0, m);
			}
		long before = freeBytes();
		for (int c = 0; c < 300; c++) {
			// Every other cycle also attends through the region, which adds its attention buffers.
			try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv);
					DeviceKvCache m = new DeviceKvCache(ctx, KV)) {
				path.run(0, x, c);
				if (c % 2 == 1)
					path.run(0, x, 0, m);
			}
		}
		assertThat(Math.abs(before - freeBytes())).as("device bytes not returned after 300 cycles")
				.isLessThanOrEqualTo(4L * 1024 * 1024);
	}

	@Test
	@DisplayName("with a mirror: k, v and the attention output match the op-at-a-time path bit for bit, at several positions")
	void attentionInsideTheRegionMatchesBitForBit() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			assertThat(path.attendsOnDevice()).as("attention kernel and FP16 cast loaded").isTrue();
			Random rnd = new Random(21);
			for (int pos : new int[] { 0, 1, 63, 64, 517, 4096 }) {
				DeviceKvCache mine = mirrorWithHistory(pos, 900 + pos);
				DeviceKvCache ref = mirrorWithHistory(pos, 900 + pos);
				try {
					float[] x = randomVec(H, rnd, -2f, 2f);
					float[][] expected = todaysPathWithAttention(x, pos, ref);
					mine.ensureCapacity(pos);
					ResidentQkvPath.Output got = path.run(0, x, pos, mine);
					assertThat(got.attended).as("attended at pos %d", pos).isTrue();
					assertThat(got.k).as("k at pos %d", pos).containsExactly(expected[1]);
					assertThat(got.v).as("v at pos %d", pos).containsExactly(expected[2]);
					assertThat(got.attn).as("attention at pos %d", pos).containsExactly(expected[3]);
					// The row is on the device, but the watermark waits for the host KV write.
					assertThat(mine.validTokens()).as("watermark before the host write").isEqualTo(pos);
					mine.markWritten(pos, 1);
					assertThat(mine.downloadK(pos + 1)).containsExactly(ref.downloadK(pos + 1));
					assertThat(mine.downloadV(pos + 1)).containsExactly(ref.downloadV(pos + 1));
				} finally {
					mine.close();
					ref.close();
				}
			}
		}
	}

	@Test
	@DisplayName("a 100-token decode through the region matches step for step and leaves the same mirror")
	void decodeRunMatchesStepForStep() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			DeviceKvCache mine = new DeviceKvCache(ctx, KV);
			DeviceKvCache ref = new DeviceKvCache(ctx, KV);
			try {
				assertThat(decodeMatches(path, mine, ref, 100, new Random(31))).isTrue();
				assertThat(mine.validTokens()).isEqualTo(100);
				assertThat(mine.downloadK(100)).containsExactly(ref.downloadK(100));
				assertThat(mine.downloadV(100)).containsExactly(ref.downloadV(100));
			} finally {
				mine.close();
				ref.close();
			}
		}
	}

	@Test
	@DisplayName("a mirror short of its history, or closed, is not attended: q, k, v come back as without one")
	void aMirrorItCannotReadIsNotAttended() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			Random rnd = new Random(41);
			DeviceKvCache shortOne = mirrorWithHistory(5, 41);
			DeviceKvCache closed = mirrorWithHistory(3, 42);
			closed.close();
			try {
				shortOne.ensureCapacity(10);
				for (DeviceKvCache m : new DeviceKvCache[] { shortOne, closed, null }) {
					int pos = 10;
					float[] x = randomVec(H, rnd, -2f, 2f);
					float[][] expected = todaysPath(x, 0, pos);
					ResidentQkvPath.Output got = path.run(0, x, pos, m);
					assertThat(got.attended).isFalse();
					assertThat(got.q).containsExactly(expected[0]);
					assertThat(got.k).containsExactly(expected[1]);
					assertThat(got.v).containsExactly(expected[2]);
				}
				assertThat(shortOne.validTokens()).as("the short mirror is not written").isEqualTo(5);
			} finally {
				shortOne.close();
			}
		}
	}

	@Test
	@DisplayName("threads decoding through the region with mirrors of their own each match")
	void concurrentDecodersEachMatch() throws Exception {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			ExecutorService pool = Executors.newFixedThreadPool(3);
			List<Future<Boolean>> results = new ArrayList<>();
			for (int t = 0; t < 3; t++) {
				long seed = 200 + t;
				results.add(pool.submit(() -> {
					DeviceKvCache mine = new DeviceKvCache(ctx, KV);
					DeviceKvCache ref = new DeviceKvCache(ctx, KV);
					try {
						return decodeMatches(path, mine, ref, 30, new Random(seed));
					} finally {
						mine.close();
						ref.close();
					}
				}));
			}
			for (Future<Boolean> f : results)
				assertThat(f.get(60, TimeUnit.SECONDS)).isTrue();
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
		}
	}

	@Test
	@DisplayName("attending through the region allocates no device memory per call")
	void attentionAllocatesNothingPerCall() {
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			DeviceKvCache mine = new DeviceKvCache(ctx, KV);
			try {
				mine.ensureCapacity(255);
				float[] x = randomVec(H, new Random(51), -2f, 2f);
				assertThat(path.run(0, x, 0, mine).attended).isTrue();
				mine.markWritten(0, 1);
				long afterFirst = freeBytes();
				long bytes = path.deviceBytes();
				for (int pos = 1; pos < 200; pos++) {
					assertThat(path.run(0, x, pos, mine).attended).isTrue();
					mine.markWritten(pos, 1);
				}
				assertThat(path.deviceBytes()).isEqualTo(bytes).isPositive();
				// Device-wide reading: another process or an earlier test's teardown can return
				// memory meanwhile (seen once in the full GPU group, +320 KiB), but an allocation
				// per call can only lower it.
				assertThat(freeBytes()).isGreaterThanOrEqualTo(afterFirst);
			} finally {
				mine.close();
			}
		}
	}

	@Test
	@DisplayName("architectures and layouts the device path cannot run are named, not silently accepted")
	void namesWhatItCannotRun() {
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.ADJACENT, false)).isNull();
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.SPLIT_HALF, false)).contains("split-half");
		assertThat(ResidentQkvPath.unsupportedReason(ctx, RopePairing.ADJACENT, true)).contains("bias");
		assertThat(ResidentQkvPath.unsupportedReason(null, RopePairing.ADJACENT, false)).contains("CUDA");
	}

	// ── The whole decode layer in the region ─────────────────────────────────

	/** Layers of the whole-layer model: 0, 1 and 3 run whole; layer 2 has no device down projection. */
	private static final int TAIL_LAYERS = 4;
	private static final int HEAD_ONLY_LAYER = 2;
	private static final int INTER = 4 * H;

	/** The four-layer model the whole-layer cases run, built on first use. */
	private static final class Whole {
		static final LlamaConfig CFG = new LlamaConfig(H, TAIL_LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER, 1000, EPS,
				THETA, "llama");
		static final float[][] ATTN_NORM = new float[TAIL_LAYERS][];
		static final float[][] FFN_NORM = new float[TAIL_LAYERS][];
		static final DeviceQ4KMatrix[] WQ = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] WK = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] WV = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] WO = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] GATE = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] UP = new DeviceQ4KMatrix[TAIL_LAYERS];
		static final DeviceQ4KMatrix[] DOWN = new DeviceQ4KMatrix[TAIL_LAYERS];
		static {
			Random rnd = new Random(77);
			for (int li = 0; li < TAIL_LAYERS; li++) {
				ATTN_NORM[li] = randomVec(H, rnd, 0.5f, 1.5f);
				FFN_NORM[li] = randomVec(H, rnd, 0.5f, 1.5f);
				WQ[li] = q4k(H, H, rnd);
				WK[li] = q4k(KV, H, rnd);
				WV[li] = q4k(KV, H, rnd);
				WO[li] = q4k(H, H, rnd);
				GATE[li] = q4k(INTER, H, rnd);
				UP[li] = q4k(INTER, H, rnd);
				// Layer 2 keeps its down projection off the device: the region runs only its attention half.
				DOWN[li] = li == HEAD_ONLY_LAYER ? null : q4k(H, INTER, rnd);
			}
		}

		static ResidentQkvPath path() {
			wholeLoaded = true;
			return ResidentQkvPath.create(ctx, CFG, ATTN_NORM, WQ, WK, WV,
					new ResidentLayerTail.Weights(FFN_NORM, WO, GATE, UP, DOWN));
		}
	}

	@Test
	@DisplayName("whole layer: the layer output, k, v and attention match the op-at-a-time path bit for bit")
	void wholeLayerMatchesBitForBit() {
		try (ResidentQkvPath path = Whole.path()) {
			assertThat(path.runsWholeLayer(0)).isTrue();
			Random rnd = new Random(61);
			for (int pos : new int[] { 0, 1, 63, 64, 517, 4096 }) {
				DeviceKvCache mine = mirrorWithHistory(pos, 700 + pos);
				DeviceKvCache ref = mirrorWithHistory(pos, 700 + pos);
				try {
					float[] x = randomVec(H, rnd, -2f, 2f);
					float[][] expected = todaysLayer(0, x, pos, ref);
					mine.ensureCapacity(pos);
					ResidentQkvPath.Output got = path.run(0, x, pos, mine);
					assertThat(got.layerDone).as("whole layer at pos %d", pos).isTrue();
					assertThat(got.k).as("k at pos %d", pos).containsExactly(expected[0]);
					assertThat(got.v).as("v at pos %d", pos).containsExactly(expected[1]);
					assertThat(got.attn).as("attention at pos %d", pos).containsExactly(expected[2]);
					assertThat(got.layer).as("layer output at pos %d", pos).containsExactly(expected[3]);
					assertThat(mine.validTokens()).as("watermark before the host write").isEqualTo(pos);
				} finally {
					mine.close();
					ref.close();
				}
			}
		}
	}

	@Test
	@DisplayName("whole layer: which layers run whole, and a call that cannot attend runs only the attention half's head")
	void wholeLayerNeedsItsWeightsAndAMirror() {
		try (ResidentQkvPath path = Whole.path()) {
			for (int li = 0; li < TAIL_LAYERS; li++)
				assertThat(path.runsWholeLayer(li)).as("layer %d", li).isEqualTo(li != HEAD_ONLY_LAYER);
			float[] x = randomVec(H, new Random(62), -2f, 2f);
			ResidentQkvPath.Output got = path.run(0, x, 3, null);
			assertThat(got.attended).isFalse();
			assertThat(got.layerDone).as("no mirror: attention and the tail stay outside").isFalse();
		}
		try (ResidentQkvPath headOnly = ResidentQkvPath.create(ctx, Whole.CFG, Whole.ATTN_NORM, Whole.WQ, Whole.WK,
				Whole.WV)) {
			assertThat(headOnly.runsWholeLayer(0)).as("no tail weights given").isFalse();
		}
	}

	@Test
	@DisplayName("whole layer: a 40-token decode through four layers in a lease matches layer for layer")
	void layersInALeaseMatchStepForStep() {
		try (ResidentQkvPath path = Whole.path()) {
			assertThat(leasedDecodeMatches(path, 40, new Random(63))).isTrue();
		}
	}

	@Test
	@DisplayName("whole layer: a layer the region never sees (declined, run on the host) makes the next one upload its input")
	void aLayerOutsideTheRegionBreaksTheChain() {
		try (ResidentQkvPath path = Whole.path()) {
			Random rnd = new Random(65);
			int pos = 9;
			DeviceKvCache[] mine = { mirrorWithHistory(pos, 65), null, mirrorWithHistory(pos, 67) };
			DeviceKvCache[] ref = { mirrorWithHistory(pos, 65), mirrorWithHistory(pos, 66), mirrorWithHistory(pos, 67) };
			try {
				float[] x = randomVec(H, rnd, -2f, 2f);
				try (ResidentQkvPath.Lease lease = path.lease()) {
					mine[0].ensureCapacity(pos);
					ResidentQkvPath.Output got = path.run(0, x, pos, mine[0], lease);
					assertThat(got.layerDone).isTrue();
					// Layer 1 as a layer whose head the region declines: the handler runs it
					// op at a time and never calls the region, so the device still holds layer 0's output.
					float[] afterLayer1 = todaysLayer(1, got.layer.clone(), pos, ref[1])[3];
					float[][] expected = todaysLayer(2, afterLayer1, pos, ref[2]);
					mine[2].ensureCapacity(pos);
					got = path.run(2, afterLayer1, pos, mine[2], lease);
					assertThat(got.k).as("layer 2 reads the row it was given").containsExactly(expected[0]);
					assertThat(got.attn).containsExactly(expected[2]);
				}
			} finally {
				for (DeviceKvCache m : mine)
					if (m != null)
						m.close();
				for (DeviceKvCache m : ref)
					m.close();
			}
		}
	}

	@Test
	@DisplayName("whole layer: threads decoding in leases of their own each match")
	void concurrentLeasesEachMatch() throws Exception {
		try (ResidentQkvPath path = Whole.path()) {
			ExecutorService pool = Executors.newFixedThreadPool(3);
			List<Future<Boolean>> results = new ArrayList<>();
			for (int t = 0; t < 3; t++) {
				long seed = 300 + t;
				results.add(pool.submit(() -> leasedDecodeMatches(path, 15, new Random(seed))));
			}
			for (Future<Boolean> f : results)
				assertThat(f.get(120, TimeUnit.SECONDS)).isTrue();
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
		}
	}

	@Test
	@DisplayName("whole layer: decoding in leases allocates no device memory per call")
	void wholeLayerAllocatesNothingPerCall() {
		try (ResidentQkvPath path = Whole.path()) {
			DeviceKvCache[] mirrors = new DeviceKvCache[TAIL_LAYERS];
			for (int li = 0; li < TAIL_LAYERS; li++) {
				mirrors[li] = new DeviceKvCache(ctx, KV);
				mirrors[li].ensureCapacity(255);
			}
			try {
				// Layers 0 and 1 only, both run whole: nothing but region work runs between
				// the two device-wide readings (layer 2's host tail would use the test's own
				// GPU reference path).
				Random rnd = new Random(64);
				wholeLayersToken(path, mirrors, 0, randomVec(H, rnd, -2f, 2f));
				long afterFirst = freeBytes();
				long bytes = path.deviceBytes();
				for (int pos = 1; pos < 100; pos++)
					wholeLayersToken(path, mirrors, pos, randomVec(H, rnd, -2f, 2f));
				assertThat(path.deviceBytes()).isEqualTo(bytes).isPositive();
				assertThat(freeBytes()).isGreaterThanOrEqualTo(afterFirst);
			} finally {
				for (DeviceKvCache m : mirrors)
					m.close();
			}
		}
	}

	/** One token through layers 0 and 1 in one lease, each layer's output feeding the next. */
	private static void wholeLayersToken(ResidentQkvPath path, DeviceKvCache[] mirrors, int pos, float[] x) {
		try (ResidentQkvPath.Lease lease = path.lease()) {
			for (int li = 0; li < 2; li++) {
				ResidentQkvPath.Output got = path.run(li, x, pos, mirrors[li], lease);
				assertThat(got.layerDone).isTrue();
				mirrors[li].markWritten(pos, 1);
				x = got.layer;
			}
		}
	}

	/**
	 * Decodes {@code steps} positions through all four layers, one lease per token,
	 * against the op-at-a-time path on mirrors of its own. True when every layer's k,
	 * v, attention and output matched bit for bit and exactly the eligible layers ran whole.
	 */
	private static boolean leasedDecodeMatches(ResidentQkvPath path, int steps, Random rnd) {
		DeviceKvCache[] mine = new DeviceKvCache[TAIL_LAYERS];
		DeviceKvCache[] ref = new DeviceKvCache[TAIL_LAYERS];
		for (int li = 0; li < TAIL_LAYERS; li++) {
			mine[li] = new DeviceKvCache(ctx, KV);
			ref[li] = new DeviceKvCache(ctx, KV);
		}
		try {
			for (int pos = 0; pos < steps; pos++) {
				float[] x = randomVec(H, rnd, -2f, 2f);
				float[] expectedX = x;
				try (ResidentQkvPath.Lease lease = path.lease()) {
					for (int li = 0; li < TAIL_LAYERS; li++) {
						float[][] expected = todaysLayer(li, expectedX, pos, ref[li]);
						mine[li].ensureCapacity(pos);
						ResidentQkvPath.Output got = path.run(li, x, pos, mine[li], lease);
						if (got.layerDone != (li != HEAD_ONLY_LAYER) || !java.util.Arrays.equals(got.k, expected[0])
								|| !java.util.Arrays.equals(got.v, expected[1])
								|| !java.util.Arrays.equals(got.attn, expected[2]))
							return false;
						mine[li].markWritten(pos, 1);
						x = got.layerDone ? got.layer : todaysTail(li, x, got.attn);
						if (!java.util.Arrays.equals(x, expected[3]))
							return false;
						expectedX = expected[3];
					}
				}
			}
			for (int li = 0; li < TAIL_LAYERS; li++)
				if (!java.util.Arrays.equals(mine[li].downloadK(steps), ref[li].downloadK(steps)))
					return false;
			return true;
		} finally {
			for (int li = 0; li < TAIL_LAYERS; li++) {
				mine[li].close();
				ref[li].close();
			}
		}
	}

	/**
	 * Today's whole decode layer {@code li} of the four-layer model, op at a time:
	 * the attention half as {@link #todaysPathWithAttention}, then {@link #todaysTail}.
	 * Returns {k, v, attention, layer output}.
	 */
	private static float[][] todaysLayer(int li, float[] x, int pos, DeviceKvCache mirror) {
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x }, Whole.ATTN_NORM[li], EPS, xn)).isTrue();
		float[][] qkv = mv.sgemvSameX(new DeviceQ4KMatrix[] { Whole.WQ[li], Whole.WK[li], Whole.WV[li] }, xn[0]);
		LlamaTransformerHandler.rope(qkv[0], pos, HEADS, HEAD_DIM, THETA);
		LlamaTransformerHandler.rope(qkv[1], pos, KV_HEADS, HEAD_DIM, THETA);
		mirror.appendToken(pos, qkv[1], qkv[2]);
		float[][] attn = new float[1][];
		assertThat(gqa.attendBatched(new DeviceKvCache[] { mirror }, new float[][] { qkv[0] }, new int[] { pos + 1 },
				attn, HEADS, HEAD_DIM, HEADS / KV_HEADS, KV)).isTrue();
		return new float[][] { qkv[1], qkv[2], attn[0], todaysTail(li, x, attn[0]) };
	}

	/**
	 * Today's rest of layer {@code li} after attention, op at a time: the K-quant
	 * GEMVs through {@link CudaMatVec}, the round-trip GPU norm, and the host's
	 * residual adds and SwiGLU. Layer 2's down projection, kept off the region, runs
	 * from the same weights through the same GEMV.
	 */
	private static float[] todaysTail(int li, float[] x, float[] attn) {
		float[] x2 = LlamaTransformerHandler.add(x, mv.sgemv(Whole.WO[li], attn));
		float[][] xn = new float[1][];
		assertThat(roundTripNorm.normalizeBatch(new float[][] { x2 }, Whole.FFN_NORM[li], EPS, xn)).isTrue();
		float[][] gu = mv.sgemvSameX(new DeviceQ4KMatrix[] { Whole.GATE[li], Whole.UP[li] }, xn[0]);
		float[] hidden = new float[INTER];
		for (int i = 0; i < INTER; i++)
			hidden[i] = LlamaTransformerHandler.silu(gu[0][i]) * gu[1][i];
		return LlamaTransformerHandler.add(x2, mv.sgemv(downOf(li), hidden));
	}

	/** Layer {@code li}'s down projection; layer 2's lives only here, off the region. */
	private static DeviceQ4KMatrix downOf(int li) {
		if (li != HEAD_ONLY_LAYER)
			return Whole.DOWN[li];
		synchronized (ResidentQkvPathTest.class) {
			if (offRegionDown == null)
				offRegionDown = q4k(H, INTER, new Random(78));
			return offRegionDown;
		}
	}

	private static DeviceQ4KMatrix offRegionDown;
	private static volatile boolean wholeLoaded;

	/**
	 * Today's decode with GPU attention after the region: q, k and v as {@link #todaysPath},
	 * the row appended to {@code mirror} through the host (FP16 pack, two copies), then the
	 * kernel through {@link CudaGqaAttention#attendBatched}. Returns {q, k, v, attention}.
	 */
	private static float[][] todaysPathWithAttention(float[] x, int pos, DeviceKvCache mirror) {
		float[][] qkv = todaysPath(x, 0, pos);
		mirror.appendToken(pos, qkv[1], qkv[2]);
		float[][] out = new float[1][];
		assertThat(gqa.attendBatched(new DeviceKvCache[] { mirror }, new float[][] { qkv[0] }, new int[] { pos + 1 },
				out, HEADS, HEAD_DIM, HEADS / KV_HEADS, KV)).isTrue();
		return new float[][] { qkv[0], qkv[1], qkv[2], out[0] };
	}

	/**
	 * Decodes {@code steps} positions from 0 through the region into {@code mine} and the
	 * op-at-a-time path into {@code ref}, marking each region row after it is read back as
	 * the handler does after its host KV write. True when every k, v and attention output
	 * matched bit for bit.
	 */
	private static boolean decodeMatches(ResidentQkvPath path, DeviceKvCache mine, DeviceKvCache ref, int steps,
			Random rnd) {
		for (int pos = 0; pos < steps; pos++) {
			float[] x = randomVec(H, rnd, -2f, 2f);
			float[][] expected = todaysPathWithAttention(x, pos, ref);
			mine.ensureCapacity(pos);
			ResidentQkvPath.Output got = path.run(0, x, pos, mine);
			if (!got.attended || !java.util.Arrays.equals(got.k, expected[1])
					|| !java.util.Arrays.equals(got.v, expected[2]) || !java.util.Arrays.equals(got.attn, expected[3]))
				return false;
			mine.markWritten(pos, 1);
		}
		return true;
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
