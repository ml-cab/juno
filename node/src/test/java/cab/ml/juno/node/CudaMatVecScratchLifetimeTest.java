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
 * Device scratch of {@link CudaMatVec} is bounded by the instance, not by the
 * threads that have ever called it. The request scheduler runs every request
 * on a new virtual thread, so scratch tied to a thread is allocated once per
 * request and never used again; nothing frees device memory on garbage
 * collection.
 *
 * <p>{@code memGetInfo} is device-wide. Run on an otherwise idle device.
 */
@Tag("gpu")
@DisplayName("CudaMatVec device scratch lifetime")
class CudaMatVecScratchLifetimeTest {

	private static final int ROWS = 512;
	private static final int COLS = 512; // two Q4_K super-blocks per row
	private static final int PREFILL = 16; // above the batched-GEMM threshold: packed batched path
	private static final int SMALL_BATCH = 4; // strided-batched GEMV path

	private static GpuContext ctx;
	private static DeviceFloatMatrix wFp32;
	private static DeviceHalfMatrix wFp16;
	private static DeviceQ4KMatrix wQ4k;
	private static float[] x;
	private static float[][] xBatch;
	private static float[][] xSmall;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		Random rnd = new Random(11);
		float[] host = randomVec(ROWS * COLS, rnd, -1f, 1f);
		wFp32 = DeviceFloatMatrix.upload(ctx, host, ROWS, COLS);
		wFp16 = DeviceHalfMatrix.uploadFromFloat32(ctx, host, ROWS, COLS);
		wQ4k = DeviceQ4KMatrix.upload(ctx, GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K), ROWS, COLS);
		x = randomVec(COLS, rnd, -2f, 2f);
		xBatch = new float[PREFILL][];
		for (int b = 0; b < PREFILL; b++)
			xBatch[b] = randomVec(COLS, rnd, -2f, 2f);
		xSmall = new float[SMALL_BATCH][];
		for (int b = 0; b < SMALL_BATCH; b++)
			xSmall[b] = randomVec(COLS, rnd, -2f, 2f);
	}

	@AfterAll
	static void destroy() {
		if (wFp32 != null)
			wFp32.close();
		if (wFp16 != null)
			wFp16.close();
		if (wQ4k != null)
			wQ4k.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("50 short-lived threads through every scratch-backed call reuse one set of scratch")
	void shortLivedThreadsShareOneScratch() throws Exception {
		CudaMatVec mv = new CudaMatVec(ctx);
		try {
			everyScratchCall(mv);
			long oneSet = mv.scratchDeviceBytes();
			assertThat(oneSet).as("the calls above allocate scratch").isPositive();
			for (int t = 0; t < 50; t++)
				runOnNewThread(() -> everyScratchCall(mv));
			assertThat(mv.scratchDeviceBytes()).as("device scratch after 50 sequential short-lived threads")
					.isEqualTo(oneSet);
		} finally {
			mv.releaseScratch();
		}
	}

	@Test
	@DisplayName("the batched-prefill scratch is not kept per thread, and holds no copy of the weight")
	void batchedScratchIsNotPerThread() throws Exception {
		CudaMatVec mv = new CudaMatVec(ctx);
		try {
			mv.sgemm(wQ4k, xBatch);
			long oneSet = mv.scratchDeviceBytes();
			assertThat(oneSet).as("the packed batched path keeps window-sized scratch, not an FP16 weight")
					.isPositive().isLessThan((long) ROWS * COLS * Short.BYTES);
			for (int t = 0; t < 50; t++)
				runOnNewThread(() -> mv.sgemm(wQ4k, xBatch));
			assertThat(mv.scratchDeviceBytes()).as("device scratch after 50 sequential short-lived threads")
					.isEqualTo(oneSet);
		} finally {
			mv.releaseScratch();
		}
	}

	@Test
	@DisplayName("device memory does not grow across 60 short-lived threads (device-wide reading)")
	void deviceMemoryFlatAcrossShortLivedThreads() throws Exception {
		// One set of scratch here is about 0.6 MB: the 512 x 512 FP16 dequant buffer
		// plus staging and a stream. Measured on the reference host (GTX 1080): with
		// the scratch kept per thread (the code before this test), 60 threads took
		// 44,040,192 bytes (42 MiB); with one set per instance, six runs read between
		// -3.0 MiB and +2.4 MiB (most of them 0) as the driver's pool settles, not
		// with the thread count. The bound sits between the two. (Those readings
		// predate the packed batched GEMM, which keeps no FP16 weight buffer, so one
		// set is now smaller; the per-thread failure mode is the same.)
		CudaMatVec mv = new CudaMatVec(ctx);
		try {
			for (int w = 0; w < 3; w++)
				runOnNewThread(() -> everyScratchCall(mv));
			long before = freeBytes();
			for (int t = 0; t < 60; t++)
				runOnNewThread(() -> everyScratchCall(mv));
			assertThat(before - freeBytes()).as("device bytes taken by 60 more short-lived threads")
					.isLessThanOrEqualTo(12L * 1024 * 1024);
		} finally {
			mv.releaseScratch();
		}
	}

	@Test
	@DisplayName("releaseScratch returns the device memory, and the next call grows it again")
	void releaseReturnsMemoryAndNextCallStillWorks() {
		CudaMatVec mv = new CudaMatVec(ctx);
		float[] reference = mv.sgemv(wQ4k, x);
		long before = freeBytes();
		everyScratchCall(mv);
		assertThat(mv.scratchDeviceBytes()).isPositive();
		mv.releaseScratch();
		assertThat(mv.scratchDeviceBytes()).isZero();
		assertThat(freeBytes()).as("free device bytes after release").isGreaterThanOrEqualTo(before);
		assertThat(mv.sgemv(wQ4k, x)).as("a released instance still computes").isEqualTo(reference);
		mv.releaseScratch();
	}

	@Test
	@DisplayName("four concurrent callers get bit-identical results to one thread")
	void concurrentCallersMatchSingleThread() throws Exception {
		CudaMatVec mv = new CudaMatVec(ctx);
		ExecutorService pool = Executors.newFixedThreadPool(4);
		try {
			Results expected = everyScratchCall(mv);
			List<Future<Results>> futures = new ArrayList<>();
			for (int t = 0; t < 4; t++)
				futures.add(pool.submit(() -> {
					Results last = null;
					for (int i = 0; i < 10; i++)
						last = everyScratchCall(mv);
					return last;
				}));
			for (Future<Results> f : futures)
				expected.assertIdentical(f.get(60, TimeUnit.SECONDS));
		} finally {
			pool.shutdown();
			assertThat(pool.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
			mv.releaseScratch();
		}
	}

	/** Every CudaMatVec entry point that uses scratch or the stream. */
	private static Results everyScratchCall(CudaMatVec mv) {
		Results r = new Results();
		r.add(mv.sgemv(wFp32, x));
		r.add(mv.sgemv(wFp16, x));
		r.add(mv.sgemv(wQ4k, x));
		r.addAll(mv.sgemvSameX(new DeviceFloatMatrix[] { wFp32, wFp32 }, x));
		r.addAll(mv.sgemvSameX(new DeviceHalfMatrix[] { wFp16, wFp16 }, x));
		r.addAll(mv.sgemvSameX(new DeviceQ4KMatrix[] { wQ4k, wQ4k }, x));
		r.addAll(mv.sgemm(wFp16, xSmall));
		r.addAll(mv.sgemm(wFp16, xBatch));
		r.addAll(mv.sgemm(wQ4k, xBatch));
		r.add(mv.sgemvTranspose(wFp32, randomVec(ROWS, new Random(3), -1f, 1f)));
		r.add(mv.sgemvTranspose(wFp16, randomVec(ROWS, new Random(4), -1f, 1f)));
		return r;
	}

	private static final class Results {
		final List<float[]> rows = new ArrayList<>();

		void add(float[] y) {
			rows.add(y);
		}

		void addAll(float[][] ys) {
			for (float[] y : ys)
				rows.add(y);
		}

		void assertIdentical(Results other) {
			assertThat(other.rows).hasSameSizeAs(rows);
			for (int i = 0; i < rows.size(); i++)
				assertThat(other.rows.get(i)).as("result %d", i).isEqualTo(rows.get(i));
		}
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
