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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The device memory kept free after the weight upload is sized from the prefill
 * window, not from a weight matrix, now that batched K-quant matmuls multiply the
 * packed weights:
 * <ul>
 * <li>the footprint the reserve and the adaptive chunk are computed from is exactly
 * what a region window allocates, at every capacity step and with a fused Q/K/V
 * projection;
 * <li>with the device full, the free-memory query the upload stop rule reads
 * reports no more than the allowance the reserve adds for memory the allocator
 * withholds;
 * <li>with only what the shrunk reserve guarantees beyond that allowance left
 * allocatable, a prefill window wider than the host path's eight rows runs both
 * packed layers on the device and gives the same result as an unconstrained run,
 * while the dequantizing route, which needs a whole FP16 weight matrix as well, runs
 * out of memory in the same place.
 * </ul>
 */
@Tag("gpu")
@DisplayName("Prefill reserve - sized from the window the packed path needs")
class PrefillReserveDeviceTest {

	private static final int H = 1024;
	/** FFN wide enough that its FP16 matrix (32 MiB) cannot fit beside the window in the reserve. */
	private static final int I = 16384;
	private static final int HEADS = 16;
	private static final int HEAD_DIM = 64;
	private static final int Q4K_BLOCK_BYTES = 144;
	private static final PrefillWindowRegion.Shape SHAPE = new PrefillWindowRegion.Shape(H, H, H, I, HEADS, HEADS,
			HEAD_DIM, 1, 1e-5f);

	private static GpuContext ctx;
	private CudaMatVec mv;
	private final List<AutoCloseable> owned = new ArrayList<>();

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
		assumeTrue(KQuantGemmKernel.tryLoad() != null, "tiled K-quant GEMM kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@BeforeEach
	void backend() {
		mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "K-quant MMQ kernel unavailable");
	}

	@AfterEach
	void release() throws Exception {
		for (int i = owned.size() - 1; i >= 0; i--)
			owned.get(i).close();
		owned.clear();
		mv.releaseScratch();
	}

	@Test
	void theFootprintIsWhatAWindowAllocates() {
		for (boolean fused : new boolean[] { false, true }) {
			PrefillWindowRegion region = fused ? region(fusedQkvLayer(1)) : region(q4kLayer(1));
			// Widening order, so each open allocates a window of its own capacity.
			for (int rows : new int[] { 9, 64, 65, 512 }) {
				try (PrefillWindowRegion.Window win = region.open(rows)) {
					assertThat(win.deviceBytes()).as("fused=" + fused + ", rows=" + rows)
							.isEqualTo(PrefillWindowFootprint.windowBytes(SHAPE, fused, rows));
				}
				assertThat(region.windowDeviceBytes(rows, rows))
						.isEqualTo(PrefillWindowFootprint.bytes(SHAPE, fused, rows, rows));
			}
		}
	}

	@Test
	void theAllocatorWithholdsNoMoreThanTheReservesAllowance() {
		// The upload stop rule reads the free-memory query; with the device full, the
		// query must not report more than the allowance the reserve adds for it.
		List<MemorySegment> all = fillDevice(0);
		try {
			assertThat(ctx.bindings().memGetInfo(ctx.deviceIndex())[0]).as("free bytes reported with the device full")
					.isLessThanOrEqualTo(DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES);
		} finally {
			for (MemorySegment s : all)
				ctx.bindings().deviceFree(s);
		}
	}

	@Test
	void aWideWindowRunsInTheShrunkReserveWhereTheDequantRouteDoesNot() {
		int w = DeviceScratchBudget.RESERVED_WINDOW_ROWS;
		PrefillWindowRegion.Layer l0 = q4kLayer(1);
		PrefillWindowRegion.Layer l1 = q4kLayer(11);
		float[][] input = rows(w, H, 3);

		// Unconstrained reference; it also loads every kernel module before memory is short.
		PrefillWindowRegion first = region(l0, l1);
		float[][] expected = copy(input);
		try (PrefillWindowRegion.Window win = first.open(w)) {
			runSplit(win, 0, expected);
			runSplit(win, 1, expected);
			win.materializeResidual(expected);
		}
		long reserve = DeviceScratchBudget.reserveBytes(first.windowDeviceBytes(w, w), 0L);
		long dequant = DeviceScratchBudget.dequantScratchBytes(H, H, I);
		assertThat(reserve).as("shrunk reserve below the dequant route's need").isLessThan(
				DeviceScratchBudget.reserveBytes(first.windowDeviceBytes(w, w), dequant));
		first.close();
		mv.releaseScratch();

		// What the reserve guarantees beyond the allocator's holdback, made allocatable and
		// nothing else: a hole of that size, the rest of the device filled, the hole freed.
		PrefillWindowRegion region = region(l0, l1);
		MemorySegment hole = ctx.bindings().deviceMalloc(ctx.deviceIndex(),
				reserve - DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES);
		List<MemorySegment> ballast = fillDevice(0);
		ctx.bindings().deviceFree(hole);
		try {
			float[][] x = copy(input);
			try (PrefillWindowRegion.Window win = region.open(w)) {
				runSplit(win, 0, x);
				runSplit(win, 1, x);
				win.materializeResidual(x);

				// Same window, same free memory: the dequantizing route also needs a whole
				// FP16 weight matrix, and there is no room for it.
				mv.dequantizeBatchedKQuant(true);
				float[][] y = copy(input);
				assertThatThrownBy(() -> runSplit(win, 0, y)).isInstanceOf(IllegalStateException.class)
						.matches(ex -> GpuLayerOffload.isVramOom((IllegalStateException) ex), "is a VRAM OOM");
				win.recoverLayerInput(y);
			} finally {
				mv.dequantizeBatchedKQuant(false);
			}
			for (int b = 0; b < w; b++)
				assertThat(x[b]).as("row " + b + " under the shrunk reserve").containsExactly(expected[b]);
		} finally {
			for (MemorySegment s : ballast)
				ctx.bindings().deviceFree(s);
		}
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	/** One layer in the split form, with the caller's attention standing in as "the output is Q". */
	private static void runSplit(PrefillWindowRegion.Window win, int li, float[][] x) {
		float[][] q = new float[x.length][H], k = new float[x.length][H], v = new float[x.length][H];
		assertThat(win.runLayer(li, x, 0, null, q, k, v)).as("split form (no RoPE on the device)").isFalse();
		win.finishLayer(li, q, x);
	}

	private PrefillWindowRegion region(PrefillWindowRegion.Layer... layers) {
		PrefillWindowRegion r = PrefillWindowRegion.create("test", mv, SHAPE, layers, null, 0f, false);
		assertThat(r).as("region").isNotNull();
		owned.add(r);
		return r;
	}

	private PrefillWindowRegion.Layer q4kLayer(long seed) {
		return PrefillWindowRegion.Layer.separate(q4k(H, H, seed), q4k(H, H, seed + 1), q4k(H, H, seed + 2),
				q4k(H, H, seed + 3), q4k(I, H, seed + 4), q4k(I, H, seed + 5), q4k(H, I, seed + 6), norm(seed + 7),
				norm(seed + 8), null, null, null);
	}

	private PrefillWindowRegion.Layer fusedQkvLayer(long seed) {
		return PrefillWindowRegion.Layer.mixed(q4k(3 * H, H, seed), null, null, null, q4k(H, H, seed + 3), null,
				q4k(I, H, seed + 4), q4k(I, H, seed + 5), q4k(H, I, seed + 6), norm(seed + 7), norm(seed + 8));
	}

	private PrefillWindowRegion.Matrix q4k(int rows, int cols, long seed) {
		DeviceQ4KMatrix m = mv.uploadQ4K(randomQ4K(rows, cols, seed), rows, cols);
		owned.add(m);
		return new PrefillWindowRegion.Matrix(m, null);
	}

	/** Allocates device memory in shrinking chunks until about {@code leave} bytes are free. */
	private static List<MemorySegment> fillDevice(long leave) {
		GpuBindings gpu = ctx.bindings();
		List<MemorySegment> held = new ArrayList<>();
		long chunk = 256L << 20;
		while (chunk >= (4L << 10)) {
			long free = gpu.memGetInfo(ctx.deviceIndex())[0];
			if (free - chunk < leave) {
				chunk >>= 1;
				continue;
			}
			try {
				held.add(gpu.deviceMalloc(ctx.deviceIndex(), chunk));
			} catch (IllegalStateException ex) {
				chunk >>= 1;
			}
		}
		return held;
	}

	private static float[] norm(long seed) {
		Random r = new Random(seed);
		float[] v = new float[H];
		for (int i = 0; i < H; i++)
			v[i] = 0.5f + r.nextFloat();
		return v;
	}

	private static float[][] rows(int n, int dim, long seed) {
		Random r = new Random(seed);
		float[][] x = new float[n][dim];
		for (float[] row : x)
			for (int i = 0; i < dim; i++)
				row[i] = r.nextFloat() * 2f - 1f;
		return x;
	}

	private static float[][] copy(float[][] x) {
		float[][] c = new float[x.length][];
		for (int b = 0; b < x.length; b++)
			c[b] = x[b].clone();
		return c;
	}

	/** Random Q4_K super-blocks with small finite FP16 scales, so the dequantized weights stay finite. */
	private static byte[] randomQ4K(int rows, int cols, long seed) {
		Random r = new Random(seed);
		int blocks = rows * (cols / 256);
		byte[] raw = new byte[blocks * Q4K_BLOCK_BYTES];
		r.nextBytes(raw);
		short scale = Float.floatToFloat16(0.01f);
		for (int b = 0; b < blocks; b++) {
			int off = b * Q4K_BLOCK_BYTES;
			raw[off] = (byte) scale;
			raw[off + 1] = (byte) (scale >> 8);
			raw[off + 2] = (byte) scale;
			raw[off + 3] = (byte) (scale >> 8);
		}
		return raw;
	}
}
