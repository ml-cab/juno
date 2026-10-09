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

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

/**
 * The context-shift rotation of FP16 K rows in place ({@code kv_shift.cu}): each
 * pair of each head turned by a per-pair cos/sin table, rounded as the host
 * rotation rounds ({@link RopeShift#back}). CUDA only; {@link #tryLoad} returns
 * {@code null} where it cannot load, and the caller rewrites the mirror from the
 * host rows instead.
 */
final class KvShiftKernel {

	private static final Logger log = Logger.getLogger(KvShiftKernel.class.getName());
	private static final String RESOURCE = "/cab/ml/juno/node/kv_shift.ptx";
	private static final int THREADS = 256;

	private static final AtomicReference<KvShiftKernel> INSTANCE = new AtomicReference<>();
	private static final ThreadLocal<KernelParams> PARAMS = ThreadLocal.withInitial(() -> new KernelParams(7));

	private final MemorySegment rotateHalf;
	@SuppressWarnings("unused")
	private final Arena moduleArena; // keeps the module and function slot alive

	private KvShiftKernel(MemorySegment rotateHalf, Arena moduleArena) {
		this.rotateHalf = rotateHalf;
		this.moduleArena = moduleArena;
	}

	/** Returns the loaded kernel, or {@code null} if the driver API or the PTX cannot load. */
	static KvShiftKernel tryLoad() {
		KvShiftKernel existing = INSTANCE.get();
		if (existing != null)
			return existing;
		synchronized (KvShiftKernel.class) {
			existing = INSTANCE.get();
			if (existing != null)
				return existing;
			if (!CudaDriverBindings.isAvailable() || !CudaAvailability.isAvailable())
				return null;
			try {
				KvShiftKernel k = loadNew();
				INSTANCE.set(k);
				log.info("GPU KV shift kernel loaded from " + RESOURCE);
				return k;
			} catch (Throwable t) {
				log.warning("GPU KV shift kernel unavailable: " + t.getMessage());
				return null;
			}
		}
	}

	private static KvShiftKernel loadNew() throws IOException {
		CudaDriverBindings drv = CudaDriverBindings.instance();
		CudaBindings cuda = CudaBindings.instance();
		CudaBindings.check(CudaBindings.callInt(cuda.cudaSetDevice, 0), "cudaSetDevice");
		// Touch the primary context so cuCtxGetCurrent succeeds.
		MemorySegment probe = cuda.deviceMalloc(0, 4);
		cuda.deviceFree(probe);

		byte[] ptxBytes;
		try (InputStream in = KvShiftKernel.class.getResourceAsStream(RESOURCE)) {
			if (in == null)
				throw new IOException("missing classpath resource " + RESOURCE);
			ptxBytes = in.readAllBytes();
		}
		Arena arena = Arena.ofShared();
		MemorySegment ptx = arena.allocate(ptxBytes.length + 1L);
		MemorySegment.copy(MemorySegment.ofArray(ptxBytes), 0, ptx, 0, ptxBytes.length);
		ptx.set(JAVA_BYTE, ptxBytes.length, (byte) 0);
		MemorySegment moduleSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleLoadData, moduleSlot, ptx),
				"cuModuleLoadData");
		MemorySegment module = moduleSlot.get(ADDRESS, 0);
		MemorySegment fnSlot = arena.allocate(ADDRESS);
		CudaDriverBindings.check(CudaDriverBindings.callInt(drv.cuModuleGetFunction, fnSlot, module,
				arena.allocateFrom("kv_rotate_half")), "cuModuleGetFunction(kv_rotate_half)");
		return new KvShiftKernel(fnSlot.get(ADDRESS, 0), arena);
	}

	/**
	 * Rotates {@code rows} FP16 K rows at {@code k} (stride {@code kvDim}): pair
	 * {@code i} of each {@code headDim}-wide head by {@code cosSin[2i]},
	 * {@code cosSin[2i + 1]} (a device array of {@code 2 * pairs} floats), pairs
	 * {@code (i, i + pairs)} when {@code splitHalf}, else {@code (2i, 2i + 1)}.
	 * Launched on the default stream.
	 */
	void rotate(MemorySegment k, long rows, int kvDim, int headDim, int pairs, boolean splitHalf,
			MemorySegment cosSin) {
		long threads = rows * (kvDim / headDim) * pairs;
		if (threads == 0)
			return;
		int grid = (int) ((threads + THREADS - 1) / THREADS);
		PARAMS.get().pointer(0, k).i64(1, rows).i32(2, kvDim).i32(3, headDim).i32(4, pairs).i32(5, splitHalf ? 1 : 0)
				.pointer(6, cosSin).launch(rotateHalf, grid, THREADS, null, "cuLaunchKernel(kv_rotate_half)");
	}
}
