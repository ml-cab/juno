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

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;

/**
 * Loads the classpath PTX module {@code prefill_window.ptx} and launches the
 * elementwise kernels of the prefill-window device region: the FP16 cast of a
 * GEMM input, SwiGLU (written as FP16 for the down projection), the residual add,
 * the Q/K/V bias add, the fused Q/K/V split, and the RMS norm in the host loop's order. See {@code prefill_window.cu} for what each computes
 * and how it matches the CPU window path.
 *
 * <p>Every launch is asynchronous on the given stream ({@code null} for the
 * default stream) and goes through a per-thread {@link KernelParams} block, so it
 * allocates nothing. One module per process; the function handles are cached.
 * Requires an active CUDA primary context, the same loading convention as
 * {@link RmsNormKernel}.
 */
final class PrefillWindowKernels {

	private static final Logger log = Logger.getLogger(PrefillWindowKernels.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/prefill_window.ptx";
	private static final int THREADS = 256;
	private static final int NORM_THREADS = 128;

	private static final AtomicReference<PrefillWindowKernels> INSTANCE = new AtomicReference<>();

	private static final ThreadLocal<KernelParams> PARAMS = ThreadLocal.withInitial(() -> new KernelParams(4));
	private static final ThreadLocal<KernelParams> NORM_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(5));
	private static final ThreadLocal<KernelParams> SPLIT_PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));

	private final MemorySegment toHalf;
	private final MemorySegment swigluHalf;
	private final MemorySegment addInPlace;
	private final MemorySegment addBias;
	private final MemorySegment rmsNormHostOrder;
	private final MemorySegment swiglu;
	private final MemorySegment residualAddBoth;
	private final MemorySegment decodeAttentionTable;
	private final MemorySegment splitQkv;
	@SuppressWarnings("unused")
	private final Arena moduleArena; // keeps the module and function slots alive

	private PrefillWindowKernels(MemorySegment toHalf, MemorySegment swigluHalf, MemorySegment addInPlace,
			MemorySegment addBias, MemorySegment rmsNormHostOrder, MemorySegment swiglu, MemorySegment residualAddBoth,
			MemorySegment decodeAttentionTable, MemorySegment splitQkv, Arena moduleArena) {
		this.toHalf = toHalf;
		this.swigluHalf = swigluHalf;
		this.addInPlace = addInPlace;
		this.addBias = addBias;
		this.rmsNormHostOrder = rmsNormHostOrder;
		this.swiglu = swiglu;
		this.residualAddBoth = residualAddBoth;
		this.decodeAttentionTable = decodeAttentionTable;
		this.splitQkv = splitQkv;
		this.moduleArena = moduleArena;
	}

	static boolean isAvailable() {
		return CudaDriverBindings.isAvailable() && CudaAvailability.isAvailable();
	}

	/** Returns the loaded kernels, or {@code null} if the driver API or the PTX cannot load. */
	static PrefillWindowKernels tryLoad() {
		PrefillWindowKernels existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (PrefillWindowKernels.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				PrefillWindowKernels k = loadNew();
				INSTANCE.set(k);
				log.info("GPU prefill-window kernels loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("GPU prefill-window kernels unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static PrefillWindowKernels loadNew() throws IOException {
		if (!CudaDriverBindings.isAvailable())
			throw new IllegalStateException("CUDA driver API unavailable");
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaBindings cuda = CudaBindings.instance();
		CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, 0), "cudaSetDevice");
		// Touch the primary context so cuCtxGetCurrent succeeds.
		MemorySegment probe = cuda.deviceMalloc(0, 4);
		cuda.deviceFree(probe);

		byte[] ptxBytes = readResource(RESOURCE);
		Arena arena = Arena.ofShared();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);

		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		return new PrefillWindowKernels(function(drv, arena, module, "to_half"),
				function(drv, arena, module, "swiglu_half"), function(drv, arena, module, "add_inplace"),
				function(drv, arena, module, "add_bias"), function(drv, arena, module, "rms_norm_host_order"),
				function(drv, arena, module, "swiglu"), function(drv, arena, module, "residual_add_both"),
				function(drv, arena, module, "decode_attention_table"), function(drv, arena, module, "split_qkv"),
				arena);
	}

	private static MemorySegment function(CudaDriverBindings drv, Arena arena, MemorySegment module, String entry) {
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, arena.allocateFrom(entry)),
				"cuModuleGetFunction(" + entry + ")");
		return fnSlot.get(ADDRESS, 0);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = PrefillWindowKernels.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/** {@code outHalf[i] = fp16(in[i])} for {@code n} floats, rounding to nearest even. */
	void toHalf(MemorySegment in, MemorySegment outHalf, long n, MemorySegment stream) {
		Objects.requireNonNull(in, "in");
		Objects.requireNonNull(outHalf, "outHalf");
		PARAMS.get().pointer(0, in).pointer(1, outHalf).i64(2, n).launch(toHalf, grid(n), THREADS, stream,
				"cuLaunchKernel(to_half)");
	}

	/**
	 * {@code outHalf[r][i] = fp16(silu(gateUp[r][i]) * gateUp[r][inter + i])} for
	 * {@code rows x inter}: the gate projection in the first {@code inter} columns
	 * of each {@code 2 * inter}-wide row of {@code gateUp}, the up projection in the
	 * second.
	 */
	void swigluToHalf(MemorySegment gateUp, MemorySegment outHalf, int rows, int inter, MemorySegment stream) {
		Objects.requireNonNull(gateUp, "gateUp");
		Objects.requireNonNull(outHalf, "outHalf");
		requirePositive(rows, inter);
		PARAMS.get().pointer(0, gateUp).pointer(1, outHalf).i32(2, rows).i32(3, inter)
				.launch(swigluHalf, grid((long) rows * inter), THREADS, stream, "cuLaunchKernel(swiglu_half)");
	}

	/**
	 * {@code out[r][i] = silu(gateUp[r][i]) * gateUp[r][inter + i]} in float, with
	 * {@link #swigluToHalf}'s layout and roundings and no FP16 cast.
	 */
	void swiglu(MemorySegment gateUp, MemorySegment out, int rows, int inter, MemorySegment stream) {
		Objects.requireNonNull(gateUp, "gateUp");
		Objects.requireNonNull(out, "out");
		requirePositive(rows, inter);
		PARAMS.get().pointer(0, gateUp).pointer(1, out).i32(2, rows).i32(3, inter)
				.launch(swiglu, grid((long) rows * inter), THREADS, stream, "cuLaunchKernel(swiglu)");
	}

	/** {@code x[i] + y[i]} into both {@code x[i]} and {@code y[i]} for {@code n} floats. */
	void residualAddBoth(MemorySegment x, MemorySegment y, long n, MemorySegment stream) {
		Objects.requireNonNull(x, "x");
		Objects.requireNonNull(y, "y");
		PARAMS.get().pointer(0, x).pointer(1, y).i64(2, n).launch(residualAddBoth, grid(n), THREADS, stream,
				"cuLaunchKernel(residual_add_both)");
	}

	/**
	 * Writes one decode row's attention table at {@code table}: the K pointer, the V
	 * pointer, then {@code seqLen} as an int, from launch arguments rather than a copy.
	 */
	void decodeAttentionTable(MemorySegment table, MemorySegment k, MemorySegment v, int seqLen,
			MemorySegment stream) {
		Objects.requireNonNull(table, "table");
		PARAMS.get().pointer(0, table).pointer(1, k).pointer(2, v).i32(3, seqLen).launch(decodeAttentionTable, 1, 1,
				stream, "cuLaunchKernel(decode_attention_table)");
	}

	/** {@code x[i] += y[i]} for {@code n} floats. */
	void addInPlace(MemorySegment x, MemorySegment y, long n, MemorySegment stream) {
		Objects.requireNonNull(x, "x");
		Objects.requireNonNull(y, "y");
		PARAMS.get().pointer(0, x).pointer(1, y).i64(2, n).launch(addInPlace, grid(n), THREADS, stream,
				"cuLaunchKernel(add_inplace)");
	}

	/**
	 * RMS norm of {@code rows} rows of {@code dim} floats with the order and roundings
	 * of {@code LlamaTransformerHandler.rmsNormInto}: bit-identical to the host norm.
	 * One block per row; {@code out} must not alias {@code x}.
	 */
	void rmsNormHostOrder(MemorySegment x, MemorySegment weight, MemorySegment out, int rows, int dim, float eps,
			MemorySegment stream) {
		Objects.requireNonNull(x, "x");
		Objects.requireNonNull(weight, "weight");
		Objects.requireNonNull(out, "out");
		requirePositive(rows, dim);
		NORM_PARAMS.get().pointer(0, x).pointer(1, weight).pointer(2, out).i32(3, dim).f32(4, eps)
				.launch(rmsNormHostOrder, rows, NORM_THREADS, stream, "cuLaunchKernel(rms_norm_host_order)");
	}

	/**
	 * Splits {@code rows} fused {@code [q; k; v]} rows ({@code qDim + 2 kvDim} floats
	 * each) into the row-major {@code q}, {@code k} and {@code v} buffers: exact copies.
	 */
	void splitQkv(MemorySegment qkv, MemorySegment q, MemorySegment k, MemorySegment v, int rows, int qDim,
			int kvDim, MemorySegment stream) {
		Objects.requireNonNull(qkv, "qkv");
		Objects.requireNonNull(q, "q");
		Objects.requireNonNull(k, "k");
		Objects.requireNonNull(v, "v");
		requirePositive(rows, qDim);
		requirePositive(rows, kvDim);
		SPLIT_PARAMS.get().pointer(0, qkv).pointer(1, q).pointer(2, k).pointer(3, v).i32(4, rows).i32(5, qDim)
				.i32(6, kvDim).launch(splitQkv, grid((long) rows * (qDim + 2L * kvDim)), THREADS, stream,
						"cuLaunchKernel(split_qkv)");
	}

	/** {@code x[r][j] += bias[j]} for {@code rows x dim}. */
	void addBias(MemorySegment x, MemorySegment bias, int rows, int dim, MemorySegment stream) {
		Objects.requireNonNull(x, "x");
		Objects.requireNonNull(bias, "bias");
		requirePositive(rows, dim);
		PARAMS.get().pointer(0, x).pointer(1, bias).i32(2, rows).i32(3, dim)
				.launch(addBias, grid((long) rows * dim), THREADS, stream, "cuLaunchKernel(add_bias)");
	}

	private static int grid(long elements) {
		if (elements <= 0)
			throw new IllegalArgumentException("element count must be positive: " + elements);
		long blocks = (elements + THREADS - 1) / THREADS;
		if (blocks > Integer.MAX_VALUE)
			throw new IllegalArgumentException("too many elements for one launch: " + elements);
		return (int) blocks;
	}

	private static void requirePositive(int rows, int width) {
		if (rows <= 0 || width <= 0)
			throw new IllegalArgumentException("rows and width must be positive: " + rows + ", " + width);
	}
}
