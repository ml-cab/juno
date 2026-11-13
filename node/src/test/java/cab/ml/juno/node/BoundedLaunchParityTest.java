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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * A window's attention and its tiled K-quant matmul split into bounded launches ({@link BoundedLaunches}) give the
 * bits one launch gives: a launch boundary falls between whole blocks, every block keeps its rows and its geometry,
 * and the GEMM's rows are independent. The budgets here are small so that small shapes split.
 */
@Tag("gpu")
@DisplayName("Bounded launches equal one launch, bit for bit")
class BoundedLaunchParityTest {

	private static GpuContext ctx;
	private static CudaBindings cuda;
	private static GqaAttentionKernel attention;
	private static KQuantGemmKernel gemm;
	private static Q4KMmqKernel pack;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		assumeTrue(GqaAttentionKernel.isAvailable(), "GPU attention kernel unavailable");
		ctx = GpuContext.init(0);
		cuda = CudaBindings.instance();
		attention = GqaAttentionKernel.tryLoad();
		gemm = KQuantGemmKernel.tryLoad();
		pack = Q4KMmqKernel.tryLoad();
		assertThat(attention).as("attention kernel").isNotNull();
		assertThat(gemm).as("tiled GEMM kernel").isNotNull();
		assertThat(pack).as("Q8_1 packing kernel").isNotNull();
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@ParameterizedTest(name = "{0}/{1}/{2} heads, {3} rows from position {4}, window {5}")
	@CsvSource({ "8, 2, 64, 700, 100, 0", "8, 2, 64, 700, 100, 96", "4, 4, 96, 333, 0, 0", "2, 1, 160, 130, 7, 0" })
	@DisplayName("prefill window over one cache")
	void prefillWindowOverOneCache(int heads, int kvHeads, int headDim, int rows, int startPos, int window) {
		int kvDim = kvHeads * headDim;
		int[] seqLens = new int[rows];
		for (int b = 0; b < rows; b++)
			seqLens[b] = startPos + b + 1;
		SplittableRandom rng = new SplittableRandom(rows * 31L + headDim);
		try (DeviceKvCache kv = new DeviceKvCache(ctx, kvDim, startPos + rows)) {
			kv.appendWindow(0, rows(rng, startPos + rows, kvDim), rows(rng, startPos + rows, kvDim), startPos + rows);
			DeviceKvCache[] caches = new DeviceKvCache[rows];
			java.util.Arrays.fill(caches, kv);
			compareAttention(caches, seqLens, heads, kvHeads, headDim, window,
					GqaAttentionKernel.rowsPerBlock(true, rows), rng);
		}
	}

	@ParameterizedTest(name = "{0} rows over five caches")
	@CsvSource({ "40", "7" })
	@DisplayName("decode rows over separate caches")
	void decodeRowsOverSeparateCaches(int rows) {
		int heads = 8, kvHeads = 2, headDim = 64, kvDim = kvHeads * headDim;
		SplittableRandom rng = new SplittableRandom(rows);
		int[] lengths = { 5, 300, 1, 77, 1024 };
		List<DeviceKvCache> owned = new ArrayList<>();
		try {
			for (int len : lengths) {
				DeviceKvCache c = new DeviceKvCache(ctx, kvDim, len);
				c.appendWindow(0, rows(rng, len, kvDim), rows(rng, len, kvDim), len);
				owned.add(c);
			}
			DeviceKvCache[] caches = new DeviceKvCache[rows];
			int[] seqLens = new int[rows];
			for (int b = 0; b < rows; b++) {
				caches[b] = owned.get(b % lengths.length);
				seqLens[b] = 1 + rng.nextInt(lengths[b % lengths.length]);
			}
			compareAttention(caches, seqLens, heads, kvHeads, headDim, 0, GqaAttentionKernel.rowsPerBlock(false, rows),
					rng);
		} finally {
			owned.forEach(DeviceKvCache::close);
		}
	}

	@ParameterizedTest(name = "type {0}, {1}x{2}, {3} rows, ldc +{4}")
	@CsvSource({ "12, 256, 512, 300, 0", "14, 320, 256, 129, 16", "13, 128, 768, 64, 0" })
	@DisplayName("tiled K-quant GEMM")
	void tiledGemm(int typeId, int rows, int cols, int batch, int ldcPad) {
		int ldc = rows + ldcPad;
		SplittableRandom rng = new SplittableRandom(typeId * 1009L + batch);
		float[] w = new float[rows * cols];
		for (int i = 0; i < w.length; i++)
			w[i] = (float) (rng.nextDouble() * 2 - 1);
		List<MemorySegment> device = new ArrayList<>();
		try (Arena host = Arena.ofConfined();
				DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, GgufKQuantCodec.encode(w, typeId), rows, cols, typeId)) {
			MemorySegment x = upload(device, randomFloats(host, rng, (long) batch * cols));
			MemorySegment q8 = malloc(device, KQuantGemmKernel.q8Bytes(batch, cols));
			long yBytes = (long) batch * ldc * Float.BYTES;
			MemorySegment one = malloc(device, yBytes);
			MemorySegment split = malloc(device, yBytes);
			pack.quantizeX(x, q8, Math.multiplyExact(batch, cols), MemorySegment.NULL);
			gemm.launch(a, q8, one, batch, ldc, MemorySegment.NULL);
			long budget = (long) BoundedLaunches.GEMM_ROW_STEP * rows * cols;
			int launches = gemm.launchBounded(a, q8, split, batch, ldc, budget, MemorySegment.NULL);
			assertThat(launches).isEqualTo((batch + BoundedLaunches.GEMM_ROW_STEP - 1) / BoundedLaunches.GEMM_ROW_STEP);
			assertSameBits(download(host, one, yBytes), download(host, split, yBytes), batch, ldc, rows);
		} finally {
			device.forEach(cuda::deviceFree);
		}
	}

	private static void compareAttention(DeviceKvCache[] caches, int[] seqLens, int heads, int kvHeads, int headDim,
			int window, int rowsPerBlock, SplittableRandom rng) {
		int rows = seqLens.length, qDim = heads * headDim, kvDim = kvHeads * headDim;
		List<MemorySegment> device = new ArrayList<>();
		try (Arena host = Arena.ofConfined()) {
			MemorySegment q = upload(device, randomFloats(host, rng, (long) rows * qDim));
			long outBytes = (long) rows * qDim * Float.BYTES;
			MemorySegment one = malloc(device, outBytes);
			MemorySegment split = malloc(device, outBytes);
			long ptrBytes = rows * ADDRESS.byteSize();
			MemorySegment tablesHost = host.allocate(2 * ptrBytes + (long) rows * Integer.BYTES);
			for (int b = 0; b < rows; b++) {
				tablesHost.setAtIndex(ADDRESS, b, caches[b].kPointer());
				tablesHost.setAtIndex(ADDRESS, rows + b, caches[b].vPointer());
				tablesHost.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, seqLens[b]);
			}
			MemorySegment tables = upload(device, tablesHost);
			MemorySegment kPtrs = tables, vPtrs = tables.asSlice(ptrBytes), lens = tables.asSlice(2 * ptrBytes);
			int gqa = heads / kvHeads;
			attention.launch(q, kPtrs, vPtrs, lens, one, rows, heads, gqa, headDim, kvDim, rowsPerBlock, window,
					MemorySegment.NULL);
			long perKey = BoundedLaunches.attentionWorkPerKey(heads, headDim);
			long budget = BoundedLaunches.attentionWork(seqLens, 0, rows, window, perKey) / 5;
			int launches = attention.launchBounded(q, kPtrs, vPtrs, lens, split, seqLens, rows, heads, gqa, headDim,
					kvDim, rowsPerBlock, window, budget, MemorySegment.NULL);
			assertThat(launches).as("the small budget splits the window").isGreaterThan(1);
			assertSameBits(download(host, one, outBytes), download(host, split, outBytes), rows, qDim, qDim);
		} finally {
			device.forEach(cuda::deviceFree);
		}
	}

	private static void assertSameBits(float[] one, float[] split, int rows, int stride, int width) {
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < width; c++) {
				int i = r * stride + c;
				if (Float.floatToRawIntBits(one[i]) != Float.floatToRawIntBits(split[i]))
					throw new AssertionError("row " + r + " column " + c + ": one launch " + one[i] + ", split "
							+ split[i]);
			}
	}

	private static float[][] rows(SplittableRandom rng, int n, int dim) {
		float[][] r = new float[n][dim];
		for (float[] row : r)
			for (int i = 0; i < dim; i++)
				row[i] = (float) (rng.nextDouble() * 2 - 1);
		return r;
	}

	private static MemorySegment randomFloats(Arena host, SplittableRandom rng, long n) {
		MemorySegment s = host.allocate(n * Float.BYTES);
		for (long i = 0; i < n; i++)
			s.setAtIndex(JAVA_FLOAT, i, (float) (rng.nextDouble() * 2 - 1));
		return s;
	}

	private static MemorySegment malloc(List<MemorySegment> device, long bytes) {
		MemorySegment d = cuda.deviceMalloc(0, bytes);
		device.add(d);
		return d;
	}

	private static MemorySegment upload(List<MemorySegment> device, MemorySegment host) {
		MemorySegment d = malloc(device, host.byteSize());
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, host.byteSize(), CudaBindings.H2D),
				"cudaMemcpy");
		return d;
	}

	private static float[] download(Arena host, MemorySegment d, long bytes) {
		MemorySegment h = host.allocate(bytes);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, h, d, bytes, CudaBindings.D2H), "cudaMemcpy");
		return h.toArray(JAVA_FLOAT);
	}
}
