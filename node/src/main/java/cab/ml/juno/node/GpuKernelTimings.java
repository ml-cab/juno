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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The device time of single launches of the longest kernels Juno runs, at the standing sweep models' shapes, read
 * by a pair of CUDA events around each launch. A display driver's watchdog (two seconds by default on Windows,
 * for a GPU that drives a display) holds each single kernel to a time limit, so the reading is per launch, not
 * per window or per token.
 *
 * <ul>
 * <li><b>Prefill attention</b> ({@link GqaAttentionKernel}): one window of {@code tokens} query rows, row
 * {@code b} over keys {@code [0, b + 1)} of one cache, as the device prefill region launches it at the start of a
 * prompt, on each sweep model's head shape.
 * <li><b>The tiled K-quant GEMM</b> ({@link KQuantGemmKernel}) over {@code tokens} activation rows, at Mistral 7B's
 * FFN shapes, the largest matrices of the sweep: gate and up 14336x4096 in Q4_K, down 4096x14336 in Q6_K, the
 * types its Q4_K_M file holds them in.
 * <li><b>The LM head</b> ({@link Q4KMmqKernel}'s packed GEMV, one row) at the sweep's largest vocabulary,
 * Qwen2.5-3B's 151936x2048, in Q6_K as every sweep model's file holds its output matrix.
 * </ul>
 *
 * <p>On the GPU the static schedule's adaptive prefill window covers a whole prompt in one window, up to a
 * device-memory ceiling, so {@code tokens} is the prompt length: the attention and GEMM launches grow with it.
 * Only the timed kernel is launched between the events (the GEMM's and GEMV's Q8_1 packing runs before them).
 * Weights and activations are seeded random values with finite scales; the kernels' time does not depend on the
 * values.
 */
public final class GpuKernelTimings {

	/** One kernel at one shape: each timed launch's device time in milliseconds, in the order taken. */
	public record Timing(String kernel, String shape, List<Double> millis) {
		public Timing {
			millis = List.copyOf(millis);
		}

		public String name() {
			return kernel + ", " + shape;
		}

		public double medianMillis() {
			List<Double> s = new ArrayList<>(millis);
			Collections.sort(s);
			return s.get(s.size() / 2);
		}

		public double maxMillis() {
			return Collections.max(millis);
		}
	}

	/** A sweep model's attention heads. */
	record AttentionShape(String model, int numHeads, int numKvHeads, int headDim) {
		int qDim() {
			return numHeads * headDim;
		}

		int kvDim() {
			return numKvHeads * headDim;
		}

		int gqaRatio() {
			return numHeads / numKvHeads;
		}
	}

	/** A weight matrix: {@code rows} outputs by {@code cols} inputs, in GGUF tensor type {@code quantType}. */
	record MatrixShape(String label, int rows, int cols, int quantType, String typeName) {
		String describe() {
			return label + " " + rows + "x" + cols + " " + typeName;
		}
	}

	static final List<AttentionShape> ATTENTION = List.of(new AttentionShape("tinyllama-1.1b", 32, 4, 64),
			new AttentionShape("qwen2.5-3b", 16, 2, 128), new AttentionShape("phi-3.5-mini", 32, 32, 96),
			new AttentionShape("mistral-7b", 32, 8, 128));

	static final List<MatrixShape> GEMM = List.of(
			new MatrixShape("mistral-7b ffn_gate and ffn_up", 14336, 4096, QuantizationLayout.TYPE_Q4_K, "Q4_K"),
			new MatrixShape("mistral-7b ffn_down", 4096, 14336, QuantizationLayout.TYPE_Q6_K, "Q6_K"));

	static final MatrixShape LM_HEAD = new MatrixShape("qwen2.5-3b output", 151936, 2048,
			QuantizationLayout.TYPE_Q6_K, "Q6_K");

	/** Untimed launches before the timed ones, so the first reading is not the clock's ramp. */
	private static final int WARMUP = 2;
	/** A finite FP16 block scale (0.0625), written over the random bytes where each block keeps its scales. */
	private static final short HALF_SCALE = 0x2C00;

	private GpuKernelTimings() {
	}

	/**
	 * Times {@code reps} single launches of each kernel, in the order listed in the class comment.
	 *
	 * @param tokens the prefill window's rows (the prompt length)
	 * @throws IllegalStateException when CUDA or one of the kernels is unavailable, with the reason
	 */
	public static List<Timing> measure(int tokens, int reps) {
		if (tokens < 1 || reps < 1)
			throw new IllegalArgumentException("tokens and reps must be >= 1 (got " + tokens + ", " + reps + ")");
		if (!CudaAvailability.isAvailable())
			throw new IllegalStateException("CUDA is not available: " + CudaAvailability.unavailableReason());
		GqaAttentionKernel attention = loaded(GqaAttentionKernel.tryLoad(), "the attention kernel");
		KQuantGemmKernel gemm = loaded(KQuantGemmKernel.tryLoad(), "the tiled K-quant GEMM kernel");
		Q4KMmqKernel gemv = loaded(Q4KMmqKernel.tryLoad(), "the packed K-quant GEMV kernel");
		CudaBindings cuda = CudaBindings.instance();
		List<Timing> out = new ArrayList<>();
		try (GpuContext ctx = GpuContext.init(0); Events events = new Events(cuda)) {
			for (AttentionShape s : ATTENTION)
				out.add(attention(ctx, cuda, events, attention, s, tokens, reps));
			for (MatrixShape m : GEMM)
				out.add(gemm(ctx, cuda, events, gemm, gemv, m, tokens, reps));
			out.add(lmHead(ctx, cuda, events, gemv, LM_HEAD, reps));
		}
		return out;
	}

	private static Timing attention(GpuContext ctx, CudaBindings cuda, Events events, GqaAttentionKernel kernel,
			AttentionShape s, int tokens, int reps) {
		if (!GqaAttentionKernel.supportsHeadDim(s.headDim()))
			throw new IllegalStateException("the attention kernel does not run head width " + s.headDim());
		SplittableRandom rng = new SplittableRandom(s.model().hashCode());
		List<MemorySegment> device = new ArrayList<>();
		try (Arena host = Arena.ofConfined(); DeviceKvCache kv = new DeviceKvCache(ctx, s.kvDim(), tokens)) {
			kv.appendWindow(0, rows(rng, tokens, s.kvDim()), rows(rng, tokens, s.kvDim()), tokens);
			MemorySegment q = upload(cuda, device, randomFloats(host, rng, (long) tokens * s.qDim()));
			MemorySegment o = malloc(cuda, device, (long) tokens * s.qDim() * Float.BYTES);
			long ptrBytes = tokens * ADDRESS.byteSize();
			MemorySegment tablesHost = host.allocate(2 * ptrBytes + (long) tokens * Integer.BYTES);
			for (int b = 0; b < tokens; b++) {
				tablesHost.setAtIndex(ADDRESS, b, kv.kPointer());
				tablesHost.setAtIndex(ADDRESS, tokens + b, kv.vPointer());
				tablesHost.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, b + 1);
			}
			MemorySegment tables = upload(cuda, device, tablesHost);
			int rowsPerBlock = GqaAttentionKernel.rowsPerBlock(true, tokens);
			List<Double> ms = events.time(() -> kernel.launch(q, tables, tables.asSlice(ptrBytes),
					tables.asSlice(2 * ptrBytes), o, tokens, s.numHeads(), s.gqaRatio(), s.headDim(), s.kvDim(),
					rowsPerBlock, 0, MemorySegment.NULL), reps);
			return new Timing("prefill attention", s.model() + " " + s.numHeads() + "/" + s.numKvHeads() + "/"
					+ s.headDim() + " heads, " + tokens + " rows over up to " + tokens + " keys", ms);
		} finally {
			device.forEach(cuda::deviceFree);
		}
	}

	private static Timing gemm(GpuContext ctx, CudaBindings cuda, Events events, KQuantGemmKernel kernel,
			Q4KMmqKernel pack, MatrixShape m, int tokens, int reps) {
		SplittableRandom rng = new SplittableRandom(m.label().hashCode());
		List<MemorySegment> device = new ArrayList<>();
		try (Arena host = Arena.ofConfined();
				DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, randomBlocks(rng, m), m.rows(), m.cols(),
						m.quantType())) {
			MemorySegment x = upload(cuda, device, randomFloats(host, rng, (long) tokens * m.cols()));
			MemorySegment q8 = malloc(cuda, device, KQuantGemmKernel.q8Bytes(tokens, m.cols()));
			MemorySegment y = malloc(cuda, device, (long) tokens * m.rows() * Float.BYTES);
			pack.quantizeX(x, q8, Math.multiplyExact(tokens, m.cols()), MemorySegment.NULL);
			List<Double> ms = events.time(() -> kernel.launch(a, q8, y, tokens, m.rows(), MemorySegment.NULL), reps);
			return new Timing("tiled K-quant GEMM", m.describe() + ", " + tokens + " rows", ms);
		} finally {
			device.forEach(cuda::deviceFree);
		}
	}

	private static Timing lmHead(GpuContext ctx, CudaBindings cuda, Events events, Q4KMmqKernel kernel,
			MatrixShape m, int reps) {
		SplittableRandom rng = new SplittableRandom(m.label().hashCode());
		List<MemorySegment> device = new ArrayList<>();
		try (Arena host = Arena.ofConfined();
				DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, randomBlocks(rng, m), m.rows(), m.cols(),
						m.quantType())) {
			MemorySegment x = upload(cuda, device, randomFloats(host, rng, m.cols()));
			MemorySegment q8 = malloc(cuda, device, Q4KMmqKernel.q8Bytes(m.cols()));
			MemorySegment y = malloc(cuda, device, (long) m.rows() * Float.BYTES);
			kernel.quantizeX(x, q8, m.cols(), MemorySegment.NULL);
			List<Double> ms = events.time(() -> kernel.launchPacked(a.devicePointer(), q8, y, m.rows(), m.cols(),
					m.quantType(), MemorySegment.NULL), reps);
			return new Timing("LM head GEMV", m.describe() + ", 1 row", ms);
		} finally {
			device.forEach(cuda::deviceFree);
		}
	}

	/** An event pair on the default stream, around one launch at a time. */
	private static final class Events implements AutoCloseable {
		private final CudaBindings cuda;
		private final Arena arena = Arena.ofConfined();
		private final MemorySegment start;
		private final MemorySegment stop;
		private final MemorySegment ms;

		Events(CudaBindings cuda) {
			this.cuda = cuda;
			this.start = create();
			this.stop = create();
			this.ms = arena.allocate(JAVA_FLOAT);
		}

		/** {@link #WARMUP} untimed launches, then {@code reps} launches, each alone between the two events. */
		List<Double> time(Runnable launch, int reps) {
			for (int i = 0; i < WARMUP; i++)
				launch.run();
			sync();
			List<Double> out = new ArrayList<>(reps);
			for (int r = 0; r < reps; r++) {
				CudaBindings.check(CudaBindings.callInt(cuda.cudaEventRecord, start, MemorySegment.NULL),
						"cudaEventRecord");
				launch.run();
				CudaBindings.check(CudaBindings.callInt(cuda.cudaEventRecord, stop, MemorySegment.NULL),
						"cudaEventRecord");
				sync();
				CudaBindings.check(CudaBindings.callInt(cuda.cudaEventElapsedTime, ms, start, stop),
						"cudaEventElapsedTime");
				out.add((double) ms.get(JAVA_FLOAT, 0));
			}
			return out;
		}

		private void sync() {
			CudaBindings.check(CudaBindings.callInt(cuda.cudaStreamSynchronize, MemorySegment.NULL),
					"cudaStreamSynchronize");
		}

		private MemorySegment create() {
			MemorySegment slot = arena.allocate(ADDRESS);
			CudaBindings.check(CudaBindings.callInt(cuda.cudaEventCreate, slot), "cudaEventCreate");
			return slot.get(ADDRESS, 0);
		}

		@Override
		public void close() {
			CudaBindings.callInt(cuda.cudaEventDestroy, start);
			CudaBindings.callInt(cuda.cudaEventDestroy, stop);
			arena.close();
		}
	}

	private static <T> T loaded(T kernel, String what) {
		if (kernel == null)
			throw new IllegalStateException(what + " did not load; the log says why");
		return kernel;
	}

	/**
	 * Random packed blocks with a finite FP16 scale in each block: Q4_K blocks (144 bytes) open with {@code d} and
	 * {@code dmin}, Q6_K blocks (210 bytes) close with {@code d}.
	 */
	private static byte[] randomBlocks(SplittableRandom rng, MatrixShape m) {
		QuantizationLayout layout = QuantizationLayout.require(m.quantType());
		byte[] raw = new byte[Math.toIntExact(layout.encodedBytes((long) m.rows() * m.cols()))];
		rng.nextBytes(raw);
		boolean q4 = m.quantType() == QuantizationLayout.TYPE_Q4_K;
		int block = q4 ? 144 : 210;
		for (int off = 0; off < raw.length; off += block) {
			if (q4) {
				putHalf(raw, off, HALF_SCALE);
				putHalf(raw, off + 2, HALF_SCALE);
			} else {
				putHalf(raw, off + 208, HALF_SCALE);
			}
		}
		return raw;
	}

	private static void putHalf(byte[] b, int off, short v) {
		b[off] = (byte) v;
		b[off + 1] = (byte) (v >>> 8);
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

	private static MemorySegment malloc(CudaBindings cuda, List<MemorySegment> device, long bytes) {
		MemorySegment d = cuda.deviceMalloc(0, bytes);
		device.add(d);
		return d;
	}

	private static MemorySegment upload(CudaBindings cuda, List<MemorySegment> device, MemorySegment host) {
		MemorySegment d = malloc(cuda, device, host.byteSize());
		CudaBindings.check(CudaBindings.callInt(cuda.cudaMemcpy, d, host, host.byteSize(), CudaBindings.H2D),
				"cudaMemcpy");
		return d;
	}
}
