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

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Panama FFI downcall handles for the CUDA Driver API ({@code libcuda}).
 *
 * <p>Used to load PTX modules and launch fused quant kernels alongside the
 * existing cudart/cuBLAS path in {@link CudaBindings}. Relies on the cudart
 * primary context (call {@code cudaSetDevice} / allocate first).
 *
 * <p>The library is found by {@link CudaLibraryLocator} ({@code libcuda.so.1} on
 * Linux, {@code nvcuda.dll} on Windows). ABI: every descriptor matches its C
 * prototype in the CUDA 12.x {@code cuda.h} on LP64 and LLP64 alike: handles and
 * pointers, {@code unsigned int} grid, block, shared-memory and flag arguments,
 * and no C {@code long}. Kernel arguments travel through {@code kernelParams} as
 * pointers to values whose widths the PTX {@code .param} declarations fix, so they
 * do not depend on the host OS.
 */
final class CudaDriverBindings {

	private static final Logger log = Logger.getLogger(CudaDriverBindings.class.getName());

	/** Where the driver library was loaded from ({@link CudaLibraryLocator.Loaded#location()}). */
	final String location;

	final MethodHandle cuInit;
	final MethodHandle cuCtxGetCurrent;
	final MethodHandle cuModuleLoadData;
	final MethodHandle cuModuleGetFunction;
	final MethodHandle cuModuleUnload;
	final MethodHandle cuLaunchKernel;

	private static final CudaDriverBindings INSTANCE;
	private static final Throwable INIT_FAILURE;

	static {
		CudaDriverBindings b = null;
		Throwable err = null;
		try {
			b = new CudaDriverBindings();
		} catch (Throwable t) {
			err = t;
			log.info("CudaDriverBindings unavailable — " + t.getMessage());
		}
		INSTANCE = b;
		INIT_FAILURE = err;
	}

	static boolean isAvailable() {
		return INSTANCE != null;
	}

	/** Why the driver library did not load (one line); null when it did. */
	static String unavailableReason() {
		return INIT_FAILURE == null ? null : INIT_FAILURE.getMessage();
	}

	static CudaDriverBindings instance() {
		if (INSTANCE == null)
			throw new IllegalStateException(
					"CUDA driver API not available: " + INIT_FAILURE.getMessage(), INIT_FAILURE);
		return INSTANCE;
	}

	private CudaDriverBindings() {
		Linker linker = Linker.nativeLinker();
		CudaLibraryLocator.Loaded driver = CudaLibraryLocator.load(CudaLibraryLocator.Library.DRIVER, 0);
		SymbolLookup cuda = driver.lookup();
		location = driver.location();

		cuInit = bind(linker, cuda, "cuInit",
				FunctionDescriptor.of(JAVA_INT, JAVA_INT));
		cuCtxGetCurrent = bind(linker, cuda, "cuCtxGetCurrent",
				FunctionDescriptor.of(JAVA_INT, ADDRESS));
		cuModuleLoadData = bind(linker, cuda, "cuModuleLoadData",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
		cuModuleGetFunction = bind(linker, cuda, "cuModuleGetFunction",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
		cuModuleUnload = bind(linker, cuda, "cuModuleUnload",
				FunctionDescriptor.of(JAVA_INT, ADDRESS));
		// CUresult cuLaunchKernel(CUfunction, gridX/Y/Z, blockX/Y/Z, shared, stream, params, extra)
		cuLaunchKernel = bind(linker, cuda, "cuLaunchKernel",
				FunctionDescriptor.of(JAVA_INT,
						ADDRESS, // f
						JAVA_INT, JAVA_INT, JAVA_INT, // grid
						JAVA_INT, JAVA_INT, JAVA_INT, // block
						JAVA_INT, // sharedMemBytes
						ADDRESS, // stream
						ADDRESS, // kernelParams
						ADDRESS)); // extra

		check(callInt(cuInit, 0), "cuInit");
		log.info("CudaDriverBindings ready — Panama FFI (" + driver.location() + ")");
	}

	static void check(int rc, String what) {
		if (rc != 0)
			throw new IllegalStateException(what + " failed: CUDA_ERROR=" + rc);
	}

	static int callInt(MethodHandle mh, Object... args) {
		try {
			return (int) mh.invokeWithArguments(args);
		} catch (Throwable t) {
			throw new IllegalStateException("native call failed: " + t.getMessage(), t);
		}
	}

	private static MethodHandle bind(Linker linker, SymbolLookup lookup, String name,
			FunctionDescriptor desc) {
		MemorySegment sym = lookup.find(name)
				.orElseThrow(() -> new UnsatisfiedLinkError("missing symbol " + name));
		return linker.downcallHandle(sym, desc);
	}
}
