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
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Kernel-only timing of the GPU attention kernel at decode width: one query
 * row over contexts from 64 to 2048 keys, on the four standing model shapes.
 * The measured launches are queued behind a prefix of long launches (2048
 * keys), so the device runs them back to back while the host is still issuing,
 * and one pair of CUDA events around them reads device time per launch, not
 * host launch latency. Prints one line per shape and context, median of five
 * repetitions of 100 launches; asserts nothing.
 *
 * <p>Lanes: {@code current}, the shipped kernel through
 * {@link GqaAttentionKernel#launch}; and, when
 * {@code -Djuno.attentionBench.oldPtx=<path>} names the PTX of the earlier
 * full-materialization kernel (entry {@code gqa_attention}, with its scores
 * scratch argument), {@code materializing}.
 *
 * <p>Run: {@code mvn test -pl node -Dgroups=gpu -Dtest=GqaAttentionDecodeBench -Djuno.attentionBench=true}.
 */
@Tag("gpu")
@DisplayName("GPU attention kernel at decode width: device time per launch")
class GqaAttentionDecodeBench {

	private static final int[] CONTEXTS = { 64, 256, 576, 1024, 2048 };
	private static final int LAUNCHES = 100;
	private static final int REPS = 5;
	/** Long launches queued ahead of each repetition so the device is never waiting on the host. */
	private static final int BUSY_PREFIX = 200;
	private static final int THREADS = 128;
	private static final int H2D = 1;

	private record Shape(String name, int numHeads, int numKvHeads, int headDim) {
		int gqaRatio() {
			return numHeads / numKvHeads;
		}

		int kvDim() {
			return numKvHeads * headDim;
		}

		int rowDim() {
			return numHeads * headDim;
		}
	}

	private static final Shape[] SHAPES = { new Shape("tinyllama 32/4/64", 32, 4, 64),
			new Shape("qwen2.5-3b 16/2/128", 16, 2, 128), new Shape("phi-3.5-mini 32/32/96", 32, 32, 96),
			new Shape("mistral-7b 32/8/128", 32, 8, 128) };

	@FunctionalInterface
	private interface Launch {
		void run(MemorySegment seqLenBuffer);
	}

	@Test
	@DisplayName("prints device microseconds per decode launch, per lane, shape and context")
	void decodeLaunchTimes() throws IOException {
		assumeTrue(Boolean.getBoolean("juno.attentionBench"), "set -Djuno.attentionBench=true to run");
		assumeTrue(CudaAvailability.isAvailable() && GqaAttentionKernel.isAvailable(), "no CUDA device");
		GqaAttentionKernel kernel = GqaAttentionKernel.tryLoad();
		assumeTrue(kernel != null, "attention kernel unavailable");
		String oldPtx = System.getProperty("juno.attentionBench.oldPtx");
		CudaBindings cuda = CudaBindings.instance();
		int maxLen = CONTEXTS[CONTEXTS.length - 1];

		try (GpuContext ctx = GpuContext.init(0); Arena arena = Arena.ofConfined()) {
			MemorySegment oldFn = oldPtx == null ? null : loadOld(arena, Files.readAllBytes(Path.of(oldPtx)));
			System.out.printf("attention decode bench: lanes current%s; us per launch, median of %d x %d%n",
					oldFn == null ? "" : ", materializing", REPS, LAUNCHES);
			for (Shape shape : SHAPES) {
				Random rng = new Random(shape.name().hashCode());
				try (DeviceKvCache kv = new DeviceKvCache(ctx, shape.kvDim())) {
					kv.appendWindow(0, rows(rng, maxLen, shape.kvDim()), rows(rng, maxLen, shape.kvDim()), maxLen);
					MemorySegment dQ = upload(cuda, arena, floats(rng, shape.rowDim()));
					MemorySegment dOut = cuda.deviceMalloc(0, (long) shape.rowDim() * Float.BYTES);
					MemorySegment dK = uploadPtr(cuda, arena, kv.kPointer());
					MemorySegment dV = uploadPtr(cuda, arena, kv.vPointer());
					MemorySegment dLen = cuda.deviceMalloc(0, Integer.BYTES);
					MemorySegment dBusy = uploadInt(cuda, arena, maxLen);
					MemorySegment dScores = cuda.deviceMalloc(0, (long) shape.numHeads() * maxLen * Float.BYTES);
					try {
						for (int seqLen : CONTEXTS) {
							MemorySegment host = arena.allocate(JAVA_INT);
							host.set(JAVA_INT, 0, seqLen);
							CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, dLen, host, (long) Integer.BYTES, H2D),
									"cudaMemcpy");
							double current = time(cuda, arena, dLen, dBusy, len -> kernel.launch(dQ, dK, dV, len, dOut, 1,
									shape.numHeads(), shape.gqaRatio(), shape.headDim(), shape.kvDim(), 1, 0, null));
							String line = String.format("%-22s ctx %5d  current %8.2f", shape.name(), seqLen, current);
							if (oldFn != null) {
								double old = time(cuda, arena, dLen, dBusy, len -> launchOld(arena, oldFn, dQ, dK, dV, len,
										dScores, dOut, shape, maxLen));
								line += String.format("  materializing %8.2f  current/materializing %.3f", old,
										current / old);
							}
							System.out.println("attention decode bench: " + line);
						}
					} finally {
						for (MemorySegment p : new MemorySegment[] { dQ, dOut, dK, dV, dLen, dBusy, dScores })
							cuda.deviceFree(p);
					}
				}
			}
		}
	}

	/**
	 * Median over {@link #REPS} of the device time of {@link #LAUNCHES} back-to-back
	 * launches at {@code len}, each repetition queued behind {@link #BUSY_PREFIX}
	 * launches at {@code busy}, in microseconds per launch.
	 */
	private static double time(CudaBindings cuda, Arena arena, MemorySegment len, MemorySegment busy, Launch launch) {
		MemorySegment start = event(cuda, arena);
		MemorySegment stop = event(cuda, arena);
		MemorySegment ms = arena.allocate(JAVA_FLOAT);
		for (int i = 0; i < BUSY_PREFIX; i++) // brings the clock up before the first cell
			launch.run(busy);
		for (int i = 0; i < 20; i++)
			launch.run(len);
		double[] reps = new double[REPS];
		for (int r = 0; r < REPS; r++) {
			for (int i = 0; i < BUSY_PREFIX; i++)
				launch.run(busy);
			CudaBindings.callInt(cuda.cudaEventRecord, start, MemorySegment.NULL);
			for (int i = 0; i < LAUNCHES; i++)
				launch.run(len);
			CudaBindings.callInt(cuda.cudaEventRecord, stop, MemorySegment.NULL);
			CudaBindings.check(CudaBindings.callInt(cuda.cudaStreamSynchronize, MemorySegment.NULL), "sync");
			CudaBindings.check(CudaBindings.callInt(cuda.cudaEventElapsedTime, ms, start, stop), "elapsed");
			reps[r] = ms.get(JAVA_FLOAT, 0) * 1000.0 / LAUNCHES;
		}
		CudaBindings.callInt(cuda.cudaEventDestroy, start);
		CudaBindings.callInt(cuda.cudaEventDestroy, stop);
		Arrays.sort(reps);
		return reps[REPS / 2];
	}

	private static MemorySegment event(CudaBindings cuda, Arena arena) {
		MemorySegment slot = arena.allocate(ADDRESS);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaEventCreate, slot), "cudaEventCreate");
		return slot.get(ADDRESS, 0);
	}

	private static MemorySegment uploadInt(CudaBindings cuda, Arena arena, int value) {
		MemorySegment d = cuda.deviceMalloc(0, Integer.BYTES);
		MemorySegment host = arena.allocateFrom(JAVA_INT, value);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, (long) Integer.BYTES, H2D), "cudaMemcpy");
		return d;
	}

	private static MemorySegment loadOld(Arena arena, byte[] ptxBytes) {
		CudaDriverBindings drv = CudaDriverBindings.instance();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);
		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx), "cuModuleLoadData");
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, moduleSlot.get(ADDRESS, 0),
				arena.allocateFrom("gqa_attention")), "cuModuleGetFunction");
		return fnSlot.get(ADDRESS, 0);
	}

	private static void launchOld(Arena arena, MemorySegment fn, MemorySegment q, MemorySegment k, MemorySegment v,
			MemorySegment len, MemorySegment scores, MemorySegment out, Shape shape, int rowStride) {
		MemorySegment params = arena.allocate(ADDRESS, 11);
		MemorySegment[] ptrs = { q, k, v, len, scores, out };
		for (int i = 0; i < ptrs.length; i++) {
			MemorySegment slot = arena.allocate(ADDRESS);
			slot.set(ADDRESS, 0, ptrs[i]);
			params.setAtIndex(ADDRESS, i, slot);
		}
		int[] ints = { shape.numHeads(), shape.gqaRatio(), shape.headDim(), shape.kvDim(), rowStride };
		for (int i = 0; i < ints.length; i++)
			params.setAtIndex(ADDRESS, 6 + i, arena.allocateFrom(JAVA_INT, ints[i]));
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuLaunchKernel, fn, shape.numHeads(), 1, 1, THREADS, 1,
				1, 0, MemorySegment.NULL, params, MemorySegment.NULL), "cuLaunchKernel(gqa_attention, materializing)");
	}

	private static MemorySegment upload(CudaBindings cuda, Arena arena, float[] values) {
		long bytes = (long) values.length * Float.BYTES;
		MemorySegment d = cuda.deviceMalloc(0, bytes);
		MemorySegment host = arena.allocate(bytes);
		MemorySegment.copy(values, 0, host, JAVA_FLOAT, 0, values.length);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, bytes, H2D), "cudaMemcpy");
		return d;
	}

	private static MemorySegment uploadPtr(CudaBindings cuda, Arena arena, MemorySegment pointer) {
		MemorySegment d = cuda.deviceMalloc(0, ADDRESS.byteSize());
		MemorySegment host = arena.allocate(ADDRESS);
		host.set(ADDRESS, 0, pointer);
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, ADDRESS.byteSize(), H2D), "cudaMemcpy");
		return d;
	}

	private static float[][] rows(Random rng, int n, int dim) {
		float[][] out = new float[n][];
		for (int i = 0; i < n; i++)
			out[i] = floats(rng, dim);
		return out;
	}

	private static float[] floats(Random rng, int n) {
		float[] out = new float[n];
		for (int i = 0; i < n; i++)
			out[i] = rng.nextFloat() * 2f - 1f;
		return out;
	}
}
