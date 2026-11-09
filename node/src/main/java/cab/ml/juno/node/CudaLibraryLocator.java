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
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Where the CUDA driver, runtime and cuBLAS libraries are looked for, per operating system.
 *
 * <ul>
 * <li>Linux: {@code libcuda.so.1}, {@code libcudart.so.12} and {@code libcublas.so.12}, then their
 * unversioned names, found by the system loader ({@code LD_LIBRARY_PATH}, the loader cache).
 * <li>Windows: {@code nvcuda.dll} from the display driver; {@code cudart64_12.dll} or
 * {@code cudart64_13.dll} and {@code cublas64_12.dll} or {@code cublas64_13.dll}, searched on
 * {@code PATH}, in {@code %CUDA_PATH%\bin} (the 12.x layout) and in {@code %CUDA_PATH%\bin\x64} (the 13.x
 * layout), then by the system loader. 12.x comes first: it is the last runtime that runs Maxwell, Pascal
 * and Volta GPUs, and every newer GPU runs it too. cuBLAS is taken at the runtime's major version
 * ({@link #forMajor}). A cuBLAS DLL found in a folder is loaded after its {@code cublasLt64_NN.dll} from
 * the same folder, because Windows looks for a DLL's dependencies on the search path, not in the folder
 * the DLL was loaded from.
 * <li>macOS and other systems: nothing; {@link GpuPlatform#cudaRefusal()} says why.
 * </ul>
 *
 * <p>{@link #candidates} and {@link #notFoundMessage} are pure functions of the platform, the environment
 * and which files exist, so every layout is unit-tested on any host. {@link #load} applies them to this
 * process.
 */
final class CudaLibraryLocator {

	/** The three CUDA libraries Juno binds. */
	enum Library {
		DRIVER("the CUDA driver library"), RUNTIME("the CUDA runtime"), BLAS("cuBLAS");

		private final String label;

		Library(String label) {
			this.label = label;
		}
	}

	/**
	 * One way to load a library.
	 *
	 * @param location a bare file name the system loader searches for, or a file found on disk
	 * @param preload  files loaded first, from the same folder (cuBLAS's {@code cublasLt} on Windows)
	 * @param major    the CUDA major version the name carries; 0 when it carries none
	 */
	record Candidate(String location, List<String> preload, int major) {
		Candidate {
			preload = List.copyOf(preload);
		}
	}

	/** A library this process loaded, where from, and its CUDA major version (0 when the name says none). */
	record Loaded(SymbolLookup lookup, String location, int major) {
	}

	/** Windows runtime majors, in the order tried. */
	private static final int[] WINDOWS_MAJORS = { 12, 13 };

	private CudaLibraryLocator() {
	}

	/** The candidates for {@code lib} on {@code os}, in the order they are tried. Empty where CUDA cannot run. */
	static List<Candidate> candidates(Library lib, GpuPlatform os, Map<String, String> env, Predicate<String> isFile) {
		return switch (os) {
			case LINUX -> linux(lib);
			case WINDOWS -> windows(lib, env, isFile);
			case MACOS, OTHER -> List.of();
		};
	}

	/** The candidates usable with a runtime of {@code requiredMajor}; all of them when it is 0. */
	static List<Candidate> forMajor(List<Candidate> candidates, int requiredMajor) {
		if (requiredMajor == 0)
			return candidates;
		return candidates.stream().filter(c -> c.major() == 0 || c.major() == requiredMajor).toList();
	}

	/**
	 * The one-line error when no candidate loads: what was looked for, and the fix on this OS. On macOS and
	 * other systems, the platform refusal.
	 */
	static String notFoundMessage(Library lib, GpuPlatform os, Map<String, String> env, int requiredMajor) {
		String refusal = os.cudaRefusal();
		if (refusal != null)
			return refusal;
		List<Candidate> named = candidates(lib, os, env, p -> false);
		List<Candidate> usable = forMajor(named, requiredMajor);
		String names = String.join(" or ", usable.stream().map(Candidate::location).distinct().toList());
		if (usable.size() < named.size())
			names += ", to match the CUDA " + requiredMajor + " runtime";
		String head = "Could not load " + lib.label + " (" + names + ")";
		if (os == GpuPlatform.LINUX)
			return head + (lib == Library.DRIVER ? ". Install the NVIDIA driver."
					: ". Install the CUDA 12.x runtime and cuBLAS, and put their library folder on LD_LIBRARY_PATH.");
		if (lib == Library.DRIVER)
			return head + ". Install the NVIDIA display driver.";
		String where = getenv(env, "CUDA_PATH") == null ? " from PATH; CUDA_PATH is not set."
				: " from PATH, %CUDA_PATH%\\bin or %CUDA_PATH%\\bin\\x64.";
		return head + where + " Install the CUDA runtime and cuBLAS (12.x for Maxwell, Pascal and Volta GPUs; "
				+ "12.x or 13.x for newer ones) and set CUDA_PATH to the CUDA installation folder.";
	}

	/**
	 * Loads {@code lib} for this process: the first candidate usable with {@code requiredMajor} (0: any) that
	 * loads.
	 *
	 * @throws IllegalStateException with the platform refusal, or {@link #notFoundMessage} when nothing loads
	 */
	static Loaded load(Library lib, int requiredMajor) {
		GpuPlatform os = GpuPlatform.current();
		os.requireCuda();
		Map<String, String> env = System.getenv();
		String foundButFailed = null;
		for (Candidate c : forMajor(candidates(lib, os, env, CudaLibraryLocator::isRegularFile), requiredMajor)) {
			for (String p : c.preload())
				open(p);
			SymbolLookup lookup = open(c.location());
			if (lookup != null)
				return new Loaded(lookup, c.location(), c.major());
			if (foundButFailed == null && isRegularFile(c.location()))
				foundButFailed = c.location();
		}
		String message = notFoundMessage(lib, os, env, requiredMajor);
		if (foundButFailed != null)
			message += " " + foundButFailed + " was found but did not load; a library it needs may be missing.";
		throw new IllegalStateException(message);
	}

	private static List<Candidate> linux(Library lib) {
		return switch (lib) {
			case DRIVER -> List.of(new Candidate("libcuda.so.1", List.of(), 0), new Candidate("libcuda.so", List.of(), 0));
			case RUNTIME -> List.of(new Candidate("libcudart.so.12", List.of(), 12),
					new Candidate("libcudart.so", List.of(), 0));
			case BLAS -> List.of(new Candidate("libcublas.so.12", List.of(), 12),
					new Candidate("libcublas.so", List.of(), 0));
		};
	}

	private static List<Candidate> windows(Library lib, Map<String, String> env, Predicate<String> isFile) {
		if (lib == Library.DRIVER)
			return List.of(new Candidate("nvcuda.dll", List.of(), 0));
		List<String> dirs = windowsSearchDirs(env);
		List<Candidate> out = new ArrayList<>();
		for (int major : WINDOWS_MAJORS) {
			String name = windowsName(lib, major);
			for (String dir : dirs) {
				String file = windowsJoin(dir, name);
				if (!isFile.test(file))
					continue;
				String lt = windowsJoin(dir, "cublasLt64_" + major + ".dll");
				List<String> preload = lib == Library.BLAS && isFile.test(lt) ? List.of(lt) : List.of();
				out.add(new Candidate(file, preload, major));
			}
		}
		for (int major : WINDOWS_MAJORS)
			out.add(new Candidate(windowsName(lib, major), List.of(), major));
		return List.copyOf(out);
	}

	private static String windowsName(Library lib, int major) {
		return (lib == Library.RUNTIME ? "cudart64_" : "cublas64_") + major + ".dll";
	}

	/** {@code PATH}'s folders in order, then {@code %CUDA_PATH%\bin} and {@code %CUDA_PATH%\bin\x64}; each once. */
	private static List<String> windowsSearchDirs(Map<String, String> env) {
		Map<String, String> dirs = new LinkedHashMap<>();
		String path = getenv(env, "PATH");
		if (path != null)
			for (String entry : path.split(";"))
				addDir(dirs, entry);
		String cudaPath = getenv(env, "CUDA_PATH");
		if (cudaPath != null) {
			String root = unquote(cudaPath);
			addDir(dirs, windowsJoin(root, "bin"));
			addDir(dirs, windowsJoin(root, "bin\\x64"));
		}
		return List.copyOf(dirs.values());
	}

	private static void addDir(Map<String, String> dirs, String entry) {
		String dir = unquote(entry);
		if (dir.isEmpty())
			return;
		String key = dir.replaceAll("[\\\\/]+$", "").toLowerCase(Locale.ROOT);
		dirs.putIfAbsent(key, dir);
	}

	private static String unquote(String s) {
		String t = s.trim();
		if (t.length() >= 2 && t.startsWith("\"") && t.endsWith("\""))
			t = t.substring(1, t.length() - 1).trim();
		return t;
	}

	private static String windowsJoin(String dir, String name) {
		return dir.endsWith("\\") || dir.endsWith("/") ? dir + name : dir + "\\" + name;
	}

	/** An environment variable, matched without regard to case as Windows does; null when unset or blank. */
	private static String getenv(Map<String, String> env, String name) {
		String value = env.get(name);
		if (value == null)
			for (Map.Entry<String, String> e : env.entrySet())
				if (e.getKey().equalsIgnoreCase(name)) {
					value = e.getValue();
					break;
				}
		return value == null || value.isBlank() ? null : value;
	}

	private static boolean isRegularFile(String location) {
		try {
			return Files.isRegularFile(Path.of(location));
		} catch (InvalidPathException e) {
			return false;
		}
	}

	private static SymbolLookup open(String location) {
		try {
			return SymbolLookup.libraryLookup(location, Arena.global());
		} catch (IllegalArgumentException e) {
			return null;
		}
	}
}
