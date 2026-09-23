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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Panama FFI downcall handles for the CUDA Driver API ({@code libcuda}).
 *
 * <p>Used to load PTX modules and launch fused quant kernels alongside the
 * existing cudart/cuBLAS path in {@link CudaBindings}. Relies on the cudart
 * primary context (call {@code cudaSetDevice} / allocate first).
 */
final class CudaDriverBindings {

	private static final Logger log = Logger.getLogger(CudaDriverBindings.class.getName());

	final MethodHandle cuInit;
	final MethodHandle cuCtxGetCurrent;
	final MethodHandle cuModuleLoadData;
	final MethodHandle cuModuleGetFunction;
	final MethodHandle cuModuleUnload;
	final MethodHandle cuLaunchKernel;
	// CUDA graph capture/replay (see CudaGraphSession).
	final MethodHandle cuStreamBeginCapture;
	final MethodHandle cuStreamEndCapture;
	final MethodHandle cuGraphInstantiate;
	final MethodHandle cuGraphLaunch;
	final MethodHandle cuGraphDestroy;
	final MethodHandle cuGraphExecDestroy;

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

	static CudaDriverBindings instance() {
		if (INSTANCE == null)
			throw new IllegalStateException(
					"CUDA driver API not available: " + INIT_FAILURE.getMessage(), INIT_FAILURE);
		return INSTANCE;
	}

	private CudaDriverBindings() {
		Linker linker = Linker.nativeLinker();
		SymbolLookup cuda = loadLibrary("libcuda.so.1", "libcuda.so");

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

		// CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode)
		cuStreamBeginCapture = bind(linker, cuda, "cuStreamBeginCapture",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
		// CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph)
		cuStreamEndCapture = bind(linker, cuda, "cuStreamEndCapture",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
		// CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags)
		cuGraphInstantiate = bind(linker, cuda, "cuGraphInstantiate",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
		// CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
		cuGraphLaunch = bind(linker, cuda, "cuGraphLaunch",
				FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
		// CUresult cuGraphDestroy(CUgraph hGraph)
		cuGraphDestroy = bind(linker, cuda, "cuGraphDestroy",
				FunctionDescriptor.of(JAVA_INT, ADDRESS));
		// CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
		cuGraphExecDestroy = bind(linker, cuda, "cuGraphExecDestroy",
				FunctionDescriptor.of(JAVA_INT, ADDRESS));

		check(callInt(cuInit, 0), "cuInit");
		log.info("CudaDriverBindings ready — Panama FFI (libcuda)");
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

	private static SymbolLookup loadLibrary(String... names) {
		Throwable last = null;
		for (String name : names) {
			try {
				return SymbolLookup.libraryLookup(name, Arena.global());
			} catch (Throwable t) {
				last = t;
			}
		}
		throw new UnsatisfiedLinkError(
				"cannot load CUDA driver library " + String.join("|", names)
						+ (last == null ? "" : ": " + last.getMessage()));
	}
}
