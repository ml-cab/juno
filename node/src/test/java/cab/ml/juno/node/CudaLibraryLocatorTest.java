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

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.CudaLibraryLocator.Candidate;
import cab.ml.juno.node.CudaLibraryLocator.Library;
import cab.ml.juno.node.CudaLibraryLocator.Loaded;

/**
 * Library resolution as a pure function of the operating system, the environment and which files
 * exist. Windows paths are plain strings here, so the Windows cases run on any host.
 */
@DisplayName("CudaLibraryLocator: CUDA library candidates per operating system")
class CudaLibraryLocatorTest {

	private static final Predicate<String> NO_FILES = p -> false;

	private static Predicate<String> files(String... paths) {
		return Set.of(paths)::contains;
	}

	private static List<String> locations(List<Candidate> cs) {
		return cs.stream().map(Candidate::location).toList();
	}

	// Linux

	@Test
	@DisplayName("Linux: the names tried are unchanged, whatever the environment and files")
	void linuxIsUnchanged() {
		Map<String, String> env = Map.of("CUDA_PATH", "/usr/local/cuda", "LD_LIBRARY_PATH", "/opt/x");
		Predicate<String> anything = p -> true;
		for (Predicate<String> isFile : List.of(NO_FILES, anything)) {
			assertThat(CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.LINUX, env, isFile))
					.containsExactly(new Candidate("libcudart.so.12", List.of(), 12),
							new Candidate("libcudart.so", List.of(), 0));
			assertThat(CudaLibraryLocator.candidates(Library.BLAS, GpuPlatform.LINUX, env, isFile))
					.containsExactly(new Candidate("libcublas.so.12", List.of(), 12),
							new Candidate("libcublas.so", List.of(), 0));
			assertThat(CudaLibraryLocator.candidates(Library.DRIVER, GpuPlatform.LINUX, env, isFile))
					.containsExactly(new Candidate("libcuda.so.1", List.of(), 0),
							new Candidate("libcuda.so", List.of(), 0));
		}
	}

	@Test
	@DisplayName("Linux: cuBLAS for a CUDA 12 runtime keeps both names, as before")
	void linuxBlasForMajor12KeepsBothNames() {
		List<Candidate> blas = CudaLibraryLocator.candidates(Library.BLAS, GpuPlatform.LINUX, Map.of(), NO_FILES);
		assertThat(locations(CudaLibraryLocator.forMajor(blas, 12)))
				.containsExactly("libcublas.so.12", "libcublas.so");
	}

	@Test
	@DisplayName("Linux: the error text names the libraries and LD_LIBRARY_PATH")
	void linuxErrorText() {
		assertThat(CudaLibraryLocator.notFoundMessage(Library.RUNTIME, GpuPlatform.LINUX, Map.of(), 0))
				.isEqualTo("Could not load the CUDA runtime (libcudart.so.12 or libcudart.so). "
						+ "Install the CUDA 12.x runtime and cuBLAS, and put their library folder on LD_LIBRARY_PATH.");
		assertThat(CudaLibraryLocator.notFoundMessage(Library.BLAS, GpuPlatform.LINUX, Map.of(), 12))
				.isEqualTo("Could not load cuBLAS (libcublas.so.12 or libcublas.so). "
						+ "Install the CUDA 12.x runtime and cuBLAS, and put their library folder on LD_LIBRARY_PATH.");
		assertThat(CudaLibraryLocator.notFoundMessage(Library.DRIVER, GpuPlatform.LINUX, Map.of(), 0))
				.isEqualTo("Could not load the CUDA driver library (libcuda.so.1 or libcuda.so). "
						+ "Install the NVIDIA driver.");
	}

	// Windows

	private static final String CUDA12 = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4";
	private static final String CUDA13 = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0";

	@Test
	@DisplayName("Windows: the driver is nvcuda.dll, found by the system loader")
	void windowsDriver() {
		assertThat(CudaLibraryLocator.candidates(Library.DRIVER, GpuPlatform.WINDOWS, Map.of("CUDA_PATH", CUDA12),
				files(CUDA12 + "\\bin\\nvcuda.dll"))).containsExactly(new Candidate("nvcuda.dll", List.of(), 0));
	}

	@Test
	@DisplayName("Windows, CUDA 12.x layout: the runtime and cuBLAS resolve from %CUDA_PATH%\\bin")
	void windowsCuda12Layout() {
		Map<String, String> env = Map.of("CUDA_PATH", CUDA12, "PATH", "C:\\Windows\\system32");
		Predicate<String> isFile = files(CUDA12 + "\\bin\\cudart64_12.dll", CUDA12 + "\\bin\\cublas64_12.dll",
				CUDA12 + "\\bin\\cublasLt64_12.dll");

		List<Candidate> runtime = CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.WINDOWS, env, isFile);
		assertThat(runtime.get(0)).isEqualTo(new Candidate(CUDA12 + "\\bin\\cudart64_12.dll", List.of(), 12));

		List<Candidate> blas = CudaLibraryLocator.candidates(Library.BLAS, GpuPlatform.WINDOWS, env, isFile);
		assertThat(blas.get(0)).as("cuBLAS loads its cublasLt companion from the same folder first")
				.isEqualTo(new Candidate(CUDA12 + "\\bin\\cublas64_12.dll",
						List.of(CUDA12 + "\\bin\\cublasLt64_12.dll"), 12));
	}

	@Test
	@DisplayName("Windows, CUDA 13.x layout: the runtime and cuBLAS resolve from %CUDA_PATH%\\bin\\x64")
	void windowsCuda13Layout() {
		Map<String, String> env = Map.of("CUDA_PATH", CUDA13);
		Predicate<String> isFile = files(CUDA13 + "\\bin\\x64\\cudart64_13.dll", CUDA13 + "\\bin\\x64\\cublas64_13.dll",
				CUDA13 + "\\bin\\x64\\cublasLt64_13.dll");

		List<Candidate> runtime = CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.WINDOWS, env, isFile);
		assertThat(runtime.get(0)).isEqualTo(new Candidate(CUDA13 + "\\bin\\x64\\cudart64_13.dll", List.of(), 13));

		List<Candidate> blas = CudaLibraryLocator.candidates(Library.BLAS, GpuPlatform.WINDOWS, env, isFile);
		assertThat(blas.get(0)).isEqualTo(new Candidate(CUDA13 + "\\bin\\x64\\cublas64_13.dll",
				List.of(CUDA13 + "\\bin\\x64\\cublasLt64_13.dll"), 13));
	}

	@Test
	@DisplayName("Windows, CUDA_PATH unset: PATH is searched (quoted entries too), then the system loader")
	void windowsCudaPathUnset() {
		Map<String, String> env = Map.of("Path", "C:\\Windows\\system32;;\"" + CUDA12 + "\\bin\";C:\\tools\\");
		List<Candidate> runtime = CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.WINDOWS, env,
				files(CUDA12 + "\\bin\\cudart64_12.dll"));
		assertThat(runtime).containsExactly(new Candidate(CUDA12 + "\\bin\\cudart64_12.dll", List.of(), 12),
				new Candidate("cudart64_12.dll", List.of(), 12), new Candidate("cudart64_13.dll", List.of(), 13));

		assertThat(CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.WINDOWS, Map.of(), NO_FILES))
				.as("no CUDA_PATH, no PATH, no files: only the system loader's own search is left")
				.containsExactly(new Candidate("cudart64_12.dll", List.of(), 12),
						new Candidate("cudart64_13.dll", List.of(), 13));
	}

	@Test
	@DisplayName("Windows: 12.x is preferred over 13.x wherever each is found")
	void windowsPrefers12() {
		Map<String, String> env = Map.of("PATH", CUDA13 + "\\bin\\x64", "CUDA_PATH", CUDA12);
		List<Candidate> runtime = CudaLibraryLocator.candidates(Library.RUNTIME, GpuPlatform.WINDOWS, env,
				files(CUDA13 + "\\bin\\x64\\cudart64_13.dll", CUDA12 + "\\bin\\cudart64_12.dll"));
		assertThat(locations(runtime)).containsExactly(CUDA12 + "\\bin\\cudart64_12.dll",
				CUDA13 + "\\bin\\x64\\cudart64_13.dll", "cudart64_12.dll", "cudart64_13.dll");
	}

	@Test
	@DisplayName("Windows: cuBLAS is taken at the runtime's major version")
	void windowsBlasMatchesTheRuntime() {
		Map<String, String> env = Map.of("PATH", CUDA13 + "\\bin\\x64", "CUDA_PATH", CUDA12);
		List<Candidate> blas = CudaLibraryLocator.candidates(Library.BLAS, GpuPlatform.WINDOWS, env,
				files(CUDA13 + "\\bin\\x64\\cublas64_13.dll", CUDA12 + "\\bin\\cublas64_12.dll"));
		assertThat(locations(CudaLibraryLocator.forMajor(blas, 13)))
				.containsExactly(CUDA13 + "\\bin\\x64\\cublas64_13.dll", "cublas64_13.dll");
		assertThat(locations(CudaLibraryLocator.forMajor(blas, 12)))
				.containsExactly(CUDA12 + "\\bin\\cublas64_12.dll", "cublas64_12.dll");
		assertThat(CudaLibraryLocator.forMajor(blas, 0)).isEqualTo(blas);
	}

	@Test
	@DisplayName("Windows: the error text names the DLLs, PATH and CUDA_PATH, and the runtime per GPU generation")
	void windowsErrorText() {
		String set = CudaLibraryLocator.notFoundMessage(Library.RUNTIME, GpuPlatform.WINDOWS,
				Map.of("CUDA_PATH", CUDA12), 0);
		assertThat(set).isEqualTo("Could not load the CUDA runtime (cudart64_12.dll or cudart64_13.dll) from PATH, "
				+ "%CUDA_PATH%\\bin or %CUDA_PATH%\\bin\\x64. Install the CUDA runtime and cuBLAS (12.x for Maxwell, "
				+ "Pascal and Volta GPUs; 12.x or 13.x for newer ones) and set CUDA_PATH to the CUDA installation folder.");

		String unset = CudaLibraryLocator.notFoundMessage(Library.RUNTIME, GpuPlatform.WINDOWS, Map.of(), 0);
		assertThat(unset).startsWith("Could not load the CUDA runtime (cudart64_12.dll or cudart64_13.dll) from PATH; "
				+ "CUDA_PATH is not set.").contains("set CUDA_PATH to the CUDA installation folder").doesNotContain("LD_LIBRARY_PATH");

		assertThat(CudaLibraryLocator.notFoundMessage(Library.BLAS, GpuPlatform.WINDOWS, Map.of("CUDA_PATH", CUDA12),
				12)).startsWith("Could not load cuBLAS (cublas64_12.dll, to match the CUDA 12 runtime) from PATH, ");

		assertThat(CudaLibraryLocator.notFoundMessage(Library.DRIVER, GpuPlatform.WINDOWS, Map.of(), 0))
				.isEqualTo("Could not load the CUDA driver library (nvcuda.dll). Install the NVIDIA display driver.");
	}

	// macOS and other systems

	@Test
	@DisplayName("macOS: no candidates, and the error text is the platform refusal")
	void macOsIsRefused() {
		for (Library lib : Library.values()) {
			assertThat(CudaLibraryLocator.candidates(lib, GpuPlatform.MACOS, Map.of("CUDA_PATH", "/x"), p -> true))
					.isEmpty();
			assertThat(CudaLibraryLocator.notFoundMessage(lib, GpuPlatform.MACOS, Map.of(), 0))
					.isEqualTo(GpuPlatform.MACOS.cudaRefusal());
			assertThat(CudaLibraryLocator.candidates(lib, GpuPlatform.OTHER, Map.of(), p -> true)).isEmpty();
		}
	}

	@Test
	@DisplayName("every error text is one line")
	void errorTextIsOneLine() {
		for (GpuPlatform os : GpuPlatform.values())
			for (Library lib : Library.values())
				for (int major : new int[] { 0, 12 })
					assertThat(CudaLibraryLocator.notFoundMessage(lib, os, Map.of(), major)).doesNotContain("\n");
	}

	// This host

	@Test
	@Tag("gpu")
	@DisplayName("on a CUDA host, the runtime and cuBLAS load through the locator at one major version")
	void loadsOnThisHost() {
		assumeTrue(CudaBindings.isAvailable(), "Skipping: no CUDA");
		Loaded runtime = CudaLibraryLocator.load(Library.RUNTIME, 0);
		Loaded blas = CudaLibraryLocator.load(Library.BLAS, runtime.major());
		assertThat(runtime.lookup().find("cudaGetDeviceCount")).isPresent();
		assertThat(blas.lookup().find("cublasCreate_v2")).isPresent();
		if (GpuPlatform.current() == GpuPlatform.LINUX) {
			assertThat(runtime.location()).isEqualTo("libcudart.so.12");
			assertThat(blas.location()).isEqualTo("libcublas.so.12");
		}
	}
}
