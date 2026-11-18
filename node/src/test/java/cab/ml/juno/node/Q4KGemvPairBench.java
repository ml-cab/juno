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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Device time of the FFN gate and up projections on each sweep model's shape: two {@code q4k_gemv} launches against
 * one {@code q4k_gemv_pair} launch, with the Q8_1 input vector at several offsets from a 64 KiB boundary (the GEMV's
 * speed moves with where that vector lands). Prints one line per shape and offset; not a pass/fail test. Opt in with
 * {@code -Djuno.gemvPairBench=true}.
 */
@Tag("gpu")
@DisplayName("Paired Q4_K gate and up GEMV bench")
class Q4KGemvPairBench {

	private static final int[] OFFSETS = { 0x0, 0x200, 0x400, 0x800, 0xe00, 0x1000, 0x1800, 0x4800, 0xbe00 };
	private static final int LAUNCHES = 50;
	private static final int REPS = 7;
	private static final int BUSY = 100;
	private static final int H2D = 1;

	private record Shape(String name, int rows, int cols) {
	}

	private static final Shape[] SHAPES = { new Shape("tinyllama", 5632, 2048), new Shape("qwen25", 11008, 2048),
			new Shape("phi35", 8192, 3072), new Shape("mistral", 14336, 4096) };

	@Test
	@DisplayName("print two launches against one paired launch per shape and input placement")
	void bench() {
		assumeTrue(Boolean.getBoolean("juno.gemvPairBench"), "set -Djuno.gemvPairBench=true to run");
		assumeTrue(CudaAvailability.isAvailable() && Q4KMmqKernel.isAvailable(), "no CUDA device");
		Q4KMmqKernel single = Q4KMmqKernel.tryLoad();
		Q4KGemvPairKernel pair = Q4KGemvPairKernel.tryLoad();
		assumeTrue(single != null && pair != null, "GEMV kernels unavailable");
		CudaBindings cuda = CudaBindings.instance();
		Random rng = new Random(5);
		try (GpuContext ctx = GpuContext.init(0); Arena arena = Arena.ofConfined()) {
			MemorySegment start = event(cuda, arena);
			MemorySegment stop = event(cuda, arena);
			MemorySegment ms = arena.allocate(JAVA_FLOAT);
			for (Shape s : SHAPES) {
				try (DeviceQ4KMatrix gate = DeviceQ4KMatrix.upload(ctx, randomQ4K(rng, s.rows(), s.cols()), s.rows(), s.cols());
						DeviceQ4KMatrix up = DeviceQ4KMatrix.upload(ctx, randomQ4K(rng, s.rows(), s.cols()), s.rows(), s.cols())) {
					long q8Bytes = Q4KMmqKernel.q8Bytes(s.cols());
					MemorySegment dX = upload(cuda, arena, floats(rng, s.cols()));
					MemorySegment area = cuda.deviceMalloc(0, 0x30000);
					MemorySegment yGate = cuda.deviceMalloc(0, (long) s.rows() * Float.BYTES);
					MemorySegment yUp = cuda.deviceMalloc(0, (long) s.rows() * Float.BYTES);
					try {
						long aligned = (area.address() + 0xFFFF) & ~0xFFFFL;
						double[] two = new double[OFFSETS.length];
						double[] one = new double[OFFSETS.length];
						for (int o = 0; o < OFFSETS.length; o++) {
							MemorySegment q8 = MemorySegment.ofAddress(aligned + OFFSETS[o]).reinterpret(q8Bytes);
							single.quantizeX(dX, q8, s.cols(), null);
							Runnable twoLaunches = () -> {
								single.launchPacked(gate.devicePointer(), q8, yGate, s.rows(), s.cols(), QuantizationLayout.TYPE_Q4_K, null);
								single.launchPacked(up.devicePointer(), q8, yUp, s.rows(), s.cols(), QuantizationLayout.TYPE_Q4_K, null);
							};
							Runnable paired = () -> pair.launch(gate.devicePointer(), up.devicePointer(), q8, yGate, yUp,
									s.rows(), s.cols(), null);
							double[] a = new double[REPS];
							double[] b = new double[REPS];
							for (int r = 0; r < REPS; r++) {
								a[r] = time(cuda, start, stop, ms, twoLaunches);
								b[r] = time(cuda, start, stop, ms, paired);
							}
							Arrays.sort(a);
							Arrays.sort(b);
							two[o] = a[REPS / 2];
							one[o] = b[REPS / 2];
							System.out.printf("GEMV-PAIR %s %dx%d offset=0x%x two_us=%.2f pair_us=%.2f ratio=%.4f%n", s.name(),
									s.rows(), s.cols(), OFFSETS[o], two[o], one[o], one[o] / two[o]);
						}
						System.out.printf("GEMV-PAIR-SPREAD %s two max/min=%.4f pair max/min=%.4f worst ratio=%.4f%n", s.name(),
								max(two) / min(two), max(one) / min(one), worstRatio(one, two));
					} finally {
						cuda.deviceFree(dX);
						cuda.deviceFree(area);
						cuda.deviceFree(yGate);
						cuda.deviceFree(yUp);
					}
				}
			}
			CudaBindings.callInt(cuda.cudaEventDestroy, start);
			CudaBindings.callInt(cuda.cudaEventDestroy, stop);
		}
	}

	/** Microseconds per call of {@code work}, over {@link #LAUNCHES} calls after {@link #BUSY} unmeasured ones. */
	private static double time(CudaBindings cuda, MemorySegment start, MemorySegment stop, MemorySegment ms, Runnable work) {
		for (int i = 0; i < BUSY; i++)
			work.run();
		CudaBindings.callInt(cuda.cudaEventRecord, start, MemorySegment.NULL);
		for (int i = 0; i < LAUNCHES; i++)
			work.run();
		CudaBindings.callInt(cuda.cudaEventRecord, stop, MemorySegment.NULL);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaStreamSynchronize, MemorySegment.NULL), "sync");
		CudaBindings.check(CudaBindings.callInt(cuda.cudaEventElapsedTime, ms, start, stop), "elapsed");
		return ms.get(JAVA_FLOAT, 0) * 1000.0 / LAUNCHES;
	}

	/** Random Q4_K blocks with a fixed finite d and dmin, so the timing never meets a NaN or infinity. */
	private static byte[] randomQ4K(Random rng, int rows, int cols) {
		int blocks = rows * (cols / Q4KMmqKernel.QK_K);
		byte[] raw = new byte[blocks * 144];
		rng.nextBytes(raw);
		for (int b = 0; b < blocks; b++) {
			int o = b * 144;
			raw[o] = 0x00; // d = dmin = 0.0625 (FP16 0x2C00), little-endian
			raw[o + 1] = 0x2C;
			raw[o + 2] = 0x00;
			raw[o + 3] = 0x2C;
		}
		return raw;
	}

	private static double worstRatio(double[] one, double[] two) {
		double w = 0;
		for (int i = 0; i < one.length; i++)
			w = Math.max(w, one[i] / two[i]);
		return w;
	}

	private static double max(double[] v) {
		return Arrays.stream(v).max().orElse(Double.NaN);
	}

	private static double min(double[] v) {
		return Arrays.stream(v).min().orElse(Double.NaN);
	}

	private static float[] floats(Random rng, int n) {
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = rng.nextFloat() * 2f - 1f;
		return v;
	}

	private static MemorySegment event(CudaBindings cuda, Arena arena) {
		MemorySegment slot = arena.allocate(ADDRESS);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaEventCreate, slot), "cudaEventCreate");
		return slot.get(ADDRESS, 0);
	}

	private static MemorySegment upload(CudaBindings cuda, Arena arena, float[] values) {
		long bytes = (long) values.length * Float.BYTES;
		MemorySegment d = cuda.deviceMalloc(0, bytes);
		MemorySegment host = arena.allocate(bytes);
		MemorySegment.copy(values, 0, host, JAVA_FLOAT, 0, values.length);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, bytes, H2D), "cudaMemcpy");
		return d;
	}
}
