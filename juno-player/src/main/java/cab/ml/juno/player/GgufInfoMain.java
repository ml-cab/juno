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
package cab.ml.juno.player;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import cab.ml.juno.node.GgufReader;

/**
 * Standalone diagnostic tool: dumps a GGUF file's full metadata and tensor
 * layout in a plain-text, copy-pasteable format.
 *
 * <p>Added 2026-07-13 after several sessions of guessing at a mmproj file's
 * actual architecture (single- vs 2-layer projector, ffn_up/ffn_down
 * orientation, projection dim) from partial log lines. This tool exists so
 * that question never has to be guessed at again: run it against a model or
 * mmproj GGUF and paste the full output for direct architecture review,
 * instead of re-deriving it from convention or trial and error.
 *
 * <p>Usage (via the {@code ./juno gguf-info} launcher subcommand, or
 * directly):
 * <pre>{@code
 * java -cp juno-player-*-shaded.jar cab.ml.juno.player.GgufInfoMain \
 *     /path/to/model.gguf [/path/to/mmproj.gguf]
 * }</pre>
 */
public final class GgufInfoMain {

	private GgufInfoMain() {
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println("Usage: GgufInfoMain <model.gguf> [mmproj.gguf]");
			System.exit(1);
		}
		dumpFile(Path.of(args[0]));
		if (args.length >= 2) {
			System.out.println();
			System.out.println("=".repeat(80));
			System.out.println();
			dumpFile(Path.of(args[1]));
		}
	}

	private static void dumpFile(Path path) throws Exception {
		System.out.println("FILE: " + path);
		try (GgufReader r = GgufReader.open(path)) {
			System.out.println();
			System.out.println("--- Metadata (" + r.allMetadata().size() + " keys, alphabetical) ---");
			Map<String, Object> sortedMeta = new TreeMap<>(r.allMetadata());
			for (Map.Entry<String, Object> e : sortedMeta.entrySet()) {
				System.out.println("  " + e.getKey() + " = " + formatValue(e.getValue()));
			}

			List<String> names = r.tensorNames();
			System.out.println();
			System.out.println("--- Tensors (" + names.size() + ", declaration order) ---");
			for (String name : names) {
				long[] dims = r.tensorDims(name);
				int type = r.tensorType(name);
				System.out.println("  " + name + "  dims=" + dimsToString(dims) + "  type=" + typeName(type) + "("
						+ type + ")");
			}
		}
	}

	private static String formatValue(Object v) {
		if (v instanceof Object[] arr) {
			int n = arr.length;
			if (n <= 16) {
				return java.util.Arrays.toString(arr);
			}
			return "array[" + n + "] (first 16: " + java.util.Arrays.toString(java.util.Arrays.copyOf(arr, 16))
					+ " ...)";
		}
		return String.valueOf(v);
	}

	private static String dimsToString(long[] dims) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < dims.length; i++) {
			if (i > 0)
				sb.append(", ");
			sb.append(dims[i]);
		}
		return sb.append("]").toString();
	}

	/**
	 * Mirrors the exact GGML_TYPE_* constants defined in {@link GgufReader} —
	 * NOT the full upstream GGML type list. A type ID appearing here as
	 * "UNKNOWN (unsupported by this reader)" is a real GGUF type this specific
	 * codebase does not know how to decode; the tensor's metadata (name, dims)
	 * is still listed correctly, but loading its data would fail.
	 */
	private static String typeName(int type) {
		return switch (type) {
			case 0 -> "F32";
			case 1 -> "F16";
			case 2 -> "Q4_0";
			case 8 -> "Q8_0";
			case 10 -> "Q2_K";
			case 11 -> "Q3_K";
			case 12 -> "Q4_K";
			case 13 -> "Q5_K";
			case 14 -> "Q6_K";
			case 30 -> "BF16";
			default -> "UNKNOWN (unsupported by this reader)";
		};
	}
}