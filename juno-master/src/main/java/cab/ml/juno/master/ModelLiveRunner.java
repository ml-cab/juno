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
package cab.ml.juno.master;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import cab.ml.juno.master.ModelLiveChecks.LiveCheck;
import cab.ml.juno.master.ModelLiveChecks.Suite;
import cab.ml.juno.node.ModelFileGate;

/**
 * Entry point of {@code ./juno test}: runs {@link ModelLiveChecks} against one real
 * model file and exits 0 when every check passed, 1 when any failed or the model
 * cannot be loaded, 2 on a usage error.
 *
 * <pre>
 *   java -cp juno-master.jar cab.ml.juno.master.ModelLiveRunner /path/to/model.gguf
 * </pre>
 *
 * {@code -DpType=pipeline|tensor|all} (default {@code all}) picks the cluster checks:
 * {@code pipeline} runs 1-6, {@code tensor} 7-8; checks 9 (in-process long-prompt
 * prefill) and 10 (in-process context shift) run in every mode. Forked nodes take {@code -Djuno.node.heap}, which the
 * launchers set from the model size.
 */
public final class ModelLiveRunner {

	private static final String USAGE = "Usage: ModelLiveRunner /path/to/model.gguf"
			+ "  (-DpType=pipeline|tensor|all, default all)";

	private ModelLiveRunner() {
	}

	public static void main(String[] args) {
		int code = run(args, System.getProperty("pType"), System.out);
		System.exit(code);
	}

	/** Runs the checks and returns the process exit code. */
	static int run(String[] args, String pType, PrintStream out) {
		if (args.length != 1 || args[0].isBlank()) {
			out.println(USAGE);
			return 2;
		}
		Set<Suite> suites;
		try {
			suites = suitesFor(pType);
		} catch (IllegalArgumentException e) {
			out.println("ERROR: " + e.getMessage());
			out.println(USAGE);
			return 2;
		}
		String modelPath = args[0];
		if (!Files.isRegularFile(Path.of(modelPath))) {
			out.println("ERROR: model file not found: " + modelPath);
			return 1;
		}
		try {
			ModelFileGate.requireLoadable(Path.of(modelPath));
			out.printf(Locale.ROOT, "Model live checks: %s (architecture %s, pType %s)%n",
					Path.of(modelPath).getFileName(), ModelLiveChecks.architecture(modelPath),
					pType == null || pType.isBlank() ? "all" : pType.strip());
			List<LiveCheck> results = ModelLiveChecks.run(modelPath, suites, out);
			long passed = results.stream().filter(LiveCheck::passed).count();
			out.printf(Locale.ROOT, "%d of %d checks passed%n", passed, results.size());
			return exitCode(results);
		} catch (IOException | RuntimeException e) {
			out.println("ERROR: " + modelPath + ": " + e.getMessage());
			return 1;
		}
	}

	/** The suites a {@code pType} value selects; checks 9 and 10 are in every one. */
	static Set<Suite> suitesFor(String pType) {
		String p = pType == null || pType.isBlank() ? "all" : pType.strip().toLowerCase(Locale.ROOT);
		return switch (p) {
		case "all" -> EnumSet.allOf(Suite.class);
		case "pipeline" -> EnumSet.of(Suite.PIPELINE, Suite.PREFILL, Suite.CONTEXT_SHIFT);
		case "tensor" -> EnumSet.of(Suite.TENSOR, Suite.PREFILL, Suite.CONTEXT_SHIFT);
		default -> throw new IllegalArgumentException("pType must be pipeline, tensor or all (got " + pType + ")");
		};
	}

	/** 0 when at least one check ran and every check passed, 1 otherwise. */
	static int exitCode(List<LiveCheck> results) {
		return !results.isEmpty() && results.stream().allMatch(LiveCheck::passed) ? 0 : 1;
	}
}
