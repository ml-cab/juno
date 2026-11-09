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

import java.util.Locale;

/**
 * The operating systems Juno's GPU backends tell apart, and which backend each one supports.
 *
 * <p>CUDA runs on Linux and Windows ({@link CudaLibraryLocator} finds its libraries per OS). ROCm runs
 * on Linux only: {@link RocmBindings} loads the Linux HIP and rocBLAS libraries. Elsewhere each backend
 * refuses with a one-line reason before any library is loaded, so the reason, not a missing-library
 * error, is what the log and an explicit {@code -Djuno.gpu.backend} choice report.
 */
enum GpuPlatform {
	LINUX, WINDOWS, MACOS, OTHER;

	/** The platform for a {@code os.name} value; {@link #OTHER} for anything unrecognised or null. */
	static GpuPlatform of(String osName) {
		String os = osName == null ? "" : osName.toLowerCase(Locale.ROOT);
		if (os.startsWith("linux"))
			return LINUX;
		if (os.startsWith("windows"))
			return WINDOWS;
		if (os.startsWith("mac"))
			return MACOS;
		return OTHER;
	}

	/** The platform this JVM runs on. */
	static GpuPlatform current() {
		return of(System.getProperty("os.name"));
	}

	/** Null when CUDA can run here; otherwise the one-line reason it cannot. */
	String cudaRefusal() {
		return switch (this) {
			case LINUX, WINDOWS -> null;
			case MACOS -> "CUDA is not available on macOS: NVIDIA's CUDA does not support it, so Juno runs on the CPU here.";
			case OTHER -> "CUDA is supported on Linux and Windows only.";
		};
	}

	/** Null when ROCm can run here; otherwise the one-line reason it cannot. */
	String rocmRefusal() {
		return switch (this) {
			case LINUX -> null;
			case WINDOWS -> "ROCm is not supported on Windows: Juno's ROCm backend runs on Linux only. "
					+ "On Windows Juno uses CUDA or the CPU.";
			case MACOS -> "ROCm is not supported on macOS: Juno's ROCm backend runs on Linux only.";
			case OTHER -> "ROCm is supported on Linux only.";
		};
	}

	/** @throws IllegalStateException with {@link #cudaRefusal()} when CUDA cannot run here */
	void requireCuda() {
		String refusal = cudaRefusal();
		if (refusal != null)
			throw new IllegalStateException(refusal);
	}

	/** @throws IllegalStateException with {@link #rocmRefusal()} when ROCm cannot run here */
	void requireRocm() {
		String refusal = rocmRefusal();
		if (refusal != null)
			throw new IllegalStateException(refusal);
	}
}
