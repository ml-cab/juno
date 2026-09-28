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
 * Loads the classpath PTX module {@code rope.ptx} and launches the GPU rotary
 * position embedding kernel ({@code rope}), which rotates a device-resident
 * activation batch in place.
 *
 * <p>Same math as {@link LlamaTransformerHandler#rope}: adjacent-pair rotation
 * within each head, row {@code r} at position {@code startPos + r}. The angle and
 * its sine and cosine are computed in double, as on the CPU, from an
 * inverse-frequency table the host computes once with the CPU path's own
 * expression ({@link #inverseFrequencies}); see {@code rope.cu} for why that
 * precision is needed and how the rotation avoids fused multiply-adds.
 *
 * <p>Split-half (NeoX) pairing, which {@link Phi3Rope} and {@link Phi2Rope}
 * apply on the CPU, is not implemented here: nothing on the device path uses it
 * yet.
 *
 * <p>One module per process; the function handle is cached. Requires an active
 * CUDA primary context, the same loading convention as {@link RmsNormKernel}.
 * Launches go through a per-thread {@link KernelParams} block, so a launch
 * allocates nothing.
 */
final class RopeKernel {

	private static final Logger log = Logger.getLogger(RopeKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/rope.ptx";
	private static final String ENTRY = "rope";
	private static final int ROPE_THREADS = 256;

	private static final AtomicReference<RopeKernel> INSTANCE = new AtomicReference<>();

	private static final ThreadLocal<KernelParams> PARAMS = ThreadLocal.withInitial(() -> new KernelParams(6));

	private final MemorySegment fn;     // CUfunction
	private final Arena moduleArena;    // keeps module/function slots alive

	private RopeKernel(MemorySegment fn, Arena moduleArena) {
		this.fn = fn;
		this.moduleArena = moduleArena;
	}

	static boolean isAvailable() {
		return CudaDriverBindings.isAvailable() && CudaAvailability.isAvailable();
	}

	/** Returns a loaded kernel, or {@code null} if the driver API / PTX cannot load. */
	static RopeKernel tryLoad() {
		RopeKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (RopeKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			try {
				RopeKernel k = loadNew();
				INSTANCE.set(k);
				log.info("GPU RoPE kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("GPU RoPE kernel unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	/**
	 * {@code 1 / theta^(2i / headDim)} for each pair index {@code i}, in double, by
	 * exactly the expression {@link LlamaTransformerHandler#rope} evaluates per
	 * pair - so the device angle starts from the same bits as the CPU angle.
	 */
	static double[] inverseFrequencies(int headDim, float ropeTheta) {
		if (headDim <= 0 || (headDim & 1) != 0)
			throw new IllegalArgumentException("headDim must be positive and even: " + headDim);
		double[] table = new double[headDim / 2];
		for (int i = 0; i < table.length; i++)
			table[i] = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
		return table;
	}

	private static RopeKernel loadNew() throws IOException {
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
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);

		MemorySegment name = arena.allocateFrom(ENTRY);
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(
				CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module, name),
				"cuModuleGetFunction(" + ENTRY + ")");
		return new RopeKernel(fnSlot.get(ADDRESS, 0), arena);
	}

	private static byte[] readResource(String path) throws IOException {
		try (InputStream in = RopeKernel.class.getResourceAsStream(path)) {
			if (in == null)
				throw new IOException("missing classpath resource " + path);
			return in.readAllBytes();
		}
	}

	/**
	 * Rotates {@code rows} rows of {@code nHeads * headDim} floats at {@code x} in
	 * place, row {@code r} at position {@code startPos + r}. {@code invFreq} is a
	 * device table of {@code headDim / 2} doubles from {@link #inverseFrequencies}.
	 * Asynchronous on {@code stream}, which may be {@code null} for the default
	 * stream.
	 */
	void launch(MemorySegment x, MemorySegment invFreq, int rows, int nHeads, int headDim, int startPos,
			MemorySegment stream) {
		Objects.requireNonNull(x, "x");
		Objects.requireNonNull(invFreq, "invFreq");
		if (rows <= 0 || nHeads <= 0 || headDim <= 0 || (headDim & 1) != 0)
			throw new IllegalArgumentException(
					"rows, nHeads and an even headDim must be positive: " + rows + ", " + nHeads + ", " + headDim);
		if (startPos < 0)
			throw new IllegalArgumentException("startPos must not be negative: " + startPos);
		long pairs = (long) rows * nHeads * (headDim / 2);
		long blocks = (pairs + ROPE_THREADS - 1) / ROPE_THREADS;
		if (blocks > Integer.MAX_VALUE)
			throw new IllegalArgumentException("too many pairs for one launch: " + pairs);
		PARAMS.get()
				.pointer(0, x)
				.pointer(1, invFreq)
				.i32(2, rows)
				.i32(3, nHeads)
				.i32(4, headDim)
				.i32(5, startPos)
				.launch(fn, (int) blocks, ROPE_THREADS, stream, "cuLaunchKernel(rope)");
	}
}
