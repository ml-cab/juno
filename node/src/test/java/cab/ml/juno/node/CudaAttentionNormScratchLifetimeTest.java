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
import java.util.concurrent.CyclicBarrier;
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
 * Device scratch of the GPU attention path ({@link CudaGqaAttention}) and the
 * round-trip GPU norm ({@link CudaRmsNorm#normalizeBatch}) is bounded by
 * concurrent callers, not by the threads that have ever called. Both run
 * outside the context's serialization lock, so concurrent callers each take
 * their own entry from a pool; a thread that ends leaves nothing behind.
 *
 * <p>{@code memGetInfo} is device-wide. Run on an otherwise idle device.
 */
@Tag("gpu")
@DisplayName("GPU attention and norm scratch lifetime")
class CudaAttentionNormScratchLifetimeTest {

	private static final int HEADS = 32;
	private static final int KV_HEADS = 4;
	private static final int HEAD_DIM = 64;
	private static final int KV_DIM = KV_HEADS * HEAD_DIM;
	private static final int SEQ = 2048;
	private static final int DIM = 4096;

	private static GpuContext ctx;
	private static DeviceKvCache kv;
	private static float[] q;
	private static float[][] xRows;
	private static float[] normWeight;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(GqaAttentionKernel.isAvailable(), "GPU attention kernel unavailable");
		ctx = GpuContext.init(0);
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
		Random rnd = new Random(21);
		kv = new DeviceKvCache(ctx, KV_DIM);
		for (int t = 0; t < SEQ; t++)
			kv.appendToken(t, randomVec(KV_DIM, rnd), randomVec(KV_DIM, rnd));
		q = randomVec(HEADS * HEAD_DIM, rnd);
		xRows = new float[32][];
		for (int b = 0; b < xRows.length; b++)
			xRows[b] = randomVec(DIM, rnd);
		normWeight = randomVec(DIM, rnd);
	}

	@AfterAll
	static void destroy() {
		if (kv != null)
			kv.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("attention: 50 short-lived threads reuse one scratch entry")
	void attentionShortLivedThreadsReuseOneEntry() throws Exception {
		CudaGqaAttention gqa = CudaGqaAttention.tryCreate(ctx);
		try {
			attend(gqa);
			long oneEntry = gqa.scratchDeviceBytes();
			assertThat(oneEntry).isPositive();
			for (int t = 0; t < 50; t++)
				runOnNewThread(() -> attend(gqa));
			assertThat(gqa.scratchDeviceBytes()).as("attention scratch after 50 short-lived threads")
					.isEqualTo(oneEntry);
		} finally {
			gqa.close();
		}
		assertThat(gqa.scratchDeviceBytes()).as("after close").isZero();
	}

	@Test
	@DisplayName("norm: 50 short-lived threads reuse one scratch entry")
	void normShortLivedThreadsReuseOneEntry() throws Exception {
		CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
		try {
			normalize(norm);
			long oneEntry = norm.scratchDeviceBytes();
			assertThat(oneEntry).isPositive();
			for (int t = 0; t < 50; t++)
				runOnNewThread(() -> normalize(norm));
			assertThat(norm.scratchDeviceBytes()).as("norm scratch after 50 short-lived threads")
					.isEqualTo(oneEntry);
		} finally {
			norm.close();
		}
		assertThat(norm.scratchDeviceBytes()).as("after close").isZero();
	}

	@Test
	@DisplayName("device memory does not grow across 60 short-lived attention threads (device-wide reading)")
	void attentionDeviceMemoryFlat() throws Exception {
		// One attention entry here is about 0.3 MB (q, out, scores over 2048
		// positions and 32 heads, pointer and length arrays) and a norm entry about
		// 1 MB of device memory (32 rows of 4096), so each thread runs both. Measured
		// on the reference host (GTX 1080): with the scratch kept per thread (the
		// code before this test), 60 threads took 79,691,776 bytes (76 MiB); pooled,
		// six runs read 0 five times and +2.4 MiB once, as the driver's pool
		// settles. The bound sits between the two.
		CudaGqaAttention gqa = CudaGqaAttention.tryCreate(ctx);
		CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
		try {
			for (int w = 0; w < 3; w++)
				runOnNewThread(() -> {
					attend(gqa);
					normalize(norm);
				});
			long before = freeBytes();
			for (int t = 0; t < 60; t++)
				runOnNewThread(() -> {
					attend(gqa);
					normalize(norm);
				});
			assertThat(before - freeBytes()).as("device bytes taken by 60 more short-lived threads")
					.isLessThanOrEqualTo(LEAK_BOUND_BYTES);
		} finally {
			gqa.close();
			norm.close();
		}
	}

	/** Between the leak-free drift and the leak, see {@link #attentionDeviceMemoryFlat}. */
	private static final long LEAK_BOUND_BYTES = 12L * 1024 * 1024;

	@Test
	@DisplayName("four concurrent callers each get results bit-identical to one thread")
	void concurrentCallersMatchSingleThread() throws Exception {
		CudaGqaAttention gqa = CudaGqaAttention.tryCreate(ctx);
		CudaRmsNorm norm = CudaRmsNorm.tryCreate(ctx);
		ExecutorService pool = Executors.newFixedThreadPool(4);
		CyclicBarrier start = new CyclicBarrier(4);
		try {
			float[] attnRef = attend(gqa);
			float[][] normRef = normalize(norm);
			List<Future<Object[]>> futures = new ArrayList<>();
			for (int t = 0; t < 4; t++)
				futures.add(pool.submit(() -> {
					start.await(10, TimeUnit.SECONDS);
					Object[] last = null;
					for (int i = 0; i < 20; i++)
						last = new Object[] { attend(gqa), normalize(norm) };
					return last;
				}));
			for (Future<Object[]> f : futures) {
				Object[] r = f.get(60, TimeUnit.SECONDS);
				assertThat((float[]) r[0]).isEqualTo(attnRef);
				assertThat((float[][]) r[1]).isDeepEqualTo(normRef);
			}
		} finally {
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
			gqa.close();
			norm.close();
		}
	}

	private static float[] attend(CudaGqaAttention gqa) {
		float[][] out = new float[1][];
		assertThat(gqa.attendBatched(new DeviceKvCache[] { kv }, new float[][] { q }, new int[] { SEQ }, out, HEADS,
				HEAD_DIM, HEADS / KV_HEADS, KV_DIM)).isTrue();
		return out[0];
	}

	private static float[][] normalize(CudaRmsNorm norm) {
		float[][] out = new float[xRows.length][];
		assertThat(norm.normalizeBatch(xRows, normWeight, 1e-5f, out)).isTrue();
		return out;
	}

	private static void runOnNewThread(Runnable r) throws InterruptedException {
		Throwable[] failure = new Throwable[1];
		Thread th = new Thread(() -> {
			try {
				r.run();
			} catch (Throwable e) {
				failure[0] = e;
			}
		});
		th.start();
		th.join();
		if (failure[0] != null)
			throw new AssertionError("call on a short-lived thread failed", failure[0]);
	}

	private static float[] randomVec(int n, Random rnd) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = rnd.nextFloat() * 2f - 1f;
		return a;
	}

	private static long freeBytes() {
		long[] info = ctx.bindings().memGetInfo(ctx.deviceIndex());
		assumeTrue(info[0] > 0, "device memory query unavailable");
		return info[0];
	}
}
