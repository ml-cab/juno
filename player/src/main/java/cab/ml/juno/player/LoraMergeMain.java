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

import cab.ml.juno.node.LoraMerge;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;

/**
 * Entry point for the {@code juno merge} subcommand.
 *
 * <p>Bakes a trained {@code .lora} adapter into a copy of the base GGUF (or
 * llamafile), producing a standalone model that no longer needs the sidecar
 * file at inference time.
 *
 * <h3>Usage</h3>
 * <pre>
 *   juno merge --model-path model.gguf --lora-path model.lora [--output merged.gguf]
 *
 *   Options:
 *     --model-path PATH    Source GGUF or llamafile (required)
 *     --lora-path PATH     Trained .lora checkpoint (required; default: &lt;model&gt;.lora)
 *     --output PATH        Destination GGUF (default: &lt;model&gt;-merged.gguf)
 *     --help               Show this message
 * </pre>
 *
 * <h3>What it does</h3>
 * <ol>
 *   <li>Copies the source file verbatim to the output path.</li>
 *   <li>For every adapter in the {@code .lora} checkpoint, dequantises the
 *       corresponding weight matrix, applies
 *       {@code W_merged = W + (alpha/rank) × B × A}, and re-quantises it
 *       back to its original format (Q4_K, Q8_0, F16, …).</li>
 *   <li>Overwrites those bytes in-place in the copy — all other content
 *       (tokeniser, metadata, un-adapted tensors) remains identical.</li>
 * </ol>
 */
public final class LoraMergeMain {

	private LoraMergeMain() {}

	public static void main(String[] args) throws Exception {
		AnsiSupport.enable();

		String modelPath  = System.getenv("MODEL_PATH");
		String loraPath   = null;
		String outputPath = null;
		boolean help      = false;

		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--model-path" -> modelPath  = args[++i];
				case "--lora-path"  -> loraPath   = args[++i];
				case "--output"     -> outputPath = args[++i];
				case "--help", "-h" -> help       = true;
				default -> {
					System.err.println("Unknown flag: " + args[i]);
					help = true;
				}
			}
		}

		if (help || modelPath == null) {
			printHelp();
			System.exit(help ? 0 : 1);
		}

		Path model = Path.of(modelPath);
		if (!Files.exists(model)) {
			err("Model not found: " + modelPath);
		}

		// Derive defaults
		if (loraPath == null) {
			loraPath = deriveLoraPath(modelPath);
		}
		if (outputPath == null) {
			outputPath = deriveMergedPath(modelPath);
		}

		Path lora   = Path.of(loraPath);
		Path output = Path.of(outputPath);

		if (!Files.exists(lora)) {
			err("LoRA checkpoint not found: " + loraPath
					+ "\n  Train one first with:  juno lora --model-path " + modelPath);
		}

		info("Merging LoRA adapter into model");
		info("  model  : " + model.toAbsolutePath());
		info("  lora   : " + lora.toAbsolutePath());
		info("  output : " + output.toAbsolutePath());
		System.out.println();

		Instant t0 = Instant.now();
		LoraMerge.Result result = LoraMerge.merge(model, lora, output);
		long ms = Duration.between(t0, Instant.now()).toMillis();

		System.out.println();
		ok("Merge complete in " + ms + " ms");
		ok("  adapters applied : " + result.adaptersApplied());
		ok("  tensors patched  : " + result.tensorsPatched().size());
		if (!result.skipped().isEmpty()) {
			System.out.println();
			warn("Skipped " + result.skipped().size() + " adapter(s):");
			result.skipped().forEach(s -> warn("  " + s));
		}
		System.out.println();
		info("Merged model written to: " + output.toAbsolutePath());
		info("Load it with:  juno local --model-path " + output);
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static void printHelp() {
		System.out.println();
		System.out.println("  juno merge — bake a LoRA adapter into a standalone GGUF");
		System.out.println();
		System.out.println("  Usage:");
		System.out.println("    juno merge --model-path model.gguf [--lora-path model.lora] [--output merged.gguf]");
		System.out.println("    MODEL_PATH=model.gguf juno merge");
		System.out.println();
		System.out.println("  Options:");
		System.out.println("    --model-path PATH    Source GGUF or llamafile (required)");
		System.out.println("    --lora-path PATH     Trained .lora checkpoint (default: <model>.lora)");
		System.out.println("    --output PATH        Output GGUF path (default: <model>-merged.gguf)");
		System.out.println("    --help               Show this message");
		System.out.println();
		System.out.println("  Example workflow:");
		System.out.println("    # 1. Fine-tune");
		System.out.println("    juno lora --model-path tinyllama.gguf");
		System.out.println("    #   /train-qa \"What is your name?\" A: \"Juno\"");
		System.out.println("    #   /save");
		System.out.println();
		System.out.println("    # 2. Merge adapter into model");
		System.out.println("    juno merge --model-path tinyllama.gguf");
		System.out.println("    # → writes tinyllama-merged.gguf");
		System.out.println();
		System.out.println("    # 3. Run the merged model — no .lora file needed");
		System.out.println("    juno local --model-path tinyllama-merged.gguf");
		System.out.println();
		System.out.println("  Supported tensor types: F32, F16, BF16, Q8_0, Q4_0,");
		System.out.println("                          Q4_K, Q5_K, Q6_K, Q2_K, Q3_K");
		System.out.println();
	}

	private static String deriveLoraPath(String modelPath) {
		// Remove any recognised extension and add .lora
		String base = modelPath;
		for (String ext : new String[]{".gguf", ".llamafile"}) {
			if (base.toLowerCase().endsWith(ext)) {
				base = base.substring(0, base.length() - ext.length());
				break;
			}
		}
		return base + ".lora";
	}

	private static String deriveMergedPath(String modelPath) {
		String base = modelPath;
		String ext  = ".gguf";
		for (String e : new String[]{".gguf", ".llamafile"}) {
			if (base.toLowerCase().endsWith(e)) {
				base = base.substring(0, base.length() - e.length());
				break;
			}
		}
		return base + "-merged" + ext;
	}

	private static void info(String msg) {
		System.out.println("\033[0;36m▶ " + msg + "\033[0m");
	}

	private static void ok(String msg) {
		System.out.println("\033[0;32m✔ " + msg + "\033[0m");
	}

	private static void warn(String msg) {
		System.out.println("\033[1;33m⚠ " + msg + "\033[0m");
	}

	private static void err(String msg) {
		System.err.println("\033[0;31m✖ " + msg + "\033[0m");
		System.exit(1);
	}
}