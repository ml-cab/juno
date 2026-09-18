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

import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.node.GgufLoraImporter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.OptionalDouble;

/**
 * Entry point for the {@code juno lora-import} subcommand.
 *
 * <p>Converts a GGUF LoRA adapter (as produced by common
 * {@code convert_lora_to_gguf.py}-style converters — see
 * {@link GgufLoraImporter}'s javadoc for the exact naming/layout convention
 * this reads, and its honest caveat about not being verified against a real
 * converter output this session) into a Juno {@code .lora} v2 checkpoint
 * usable with {@code --lora-play}.
 *
 * <h3>Usage</h3>
 * <pre>
 *   juno lora-import --gguf adapter.gguf --out x.lora [--alpha N]
 *
 *   Options:
 *     --gguf PATH    Source GGUF LoRA adapter (required)
 *     --out PATH     Destination .lora v2 checkpoint (required)
 *     --alpha N      Override alpha for every imported adapter (default:
 *                    GGUF "adapter.lora.alpha" metadata if present, else
 *                    alpha == rank i.e. scale 1.0)
 *     --help         Show this message
 * </pre>
 */
public final class LoraImportMain {

	private LoraImportMain() {
	}

	public static void main(String[] args) throws Exception {
		AnsiSupport.enable();

		String ggufPath = null;
		String outPath = null;
		OptionalDouble alpha = OptionalDouble.empty();
		boolean help = false;

		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--gguf" -> ggufPath = args[++i];
			case "--out" -> outPath = args[++i];
			case "--alpha" -> alpha = OptionalDouble.of(Double.parseDouble(args[++i]));
			case "--help", "-h" -> help = true;
			default -> {
				System.err.println("Unknown flag: " + args[i]);
				help = true;
			}
			}
		}

		if (help || ggufPath == null || outPath == null) {
			printHelp();
			System.exit(help ? 0 : 1);
			return;
		}

		Path gguf = Path.of(ggufPath);
		if (!Files.exists(gguf))
			err("GGUF adapter not found: " + ggufPath);

		Path out = Path.of(outPath);

		info("Importing GGUF LoRA adapter");
		info("  gguf  : " + gguf.toAbsolutePath());
		info("  out   : " + out.toAbsolutePath());
		if (alpha.isPresent())
			info("  alpha : " + alpha.getAsDouble() + " (explicit override)");
		System.out.println();

		Instant t0 = Instant.now();
		LoraAdapterSet set;
		try {
			set = GgufLoraImporter.importFrom(gguf, alpha);
		} catch (IllegalArgumentException e) {
			err("Import failed: " + e.getMessage());
			return; // unreachable — err() exits
		}
		set.save(out);
		long ms = Duration.between(t0, Instant.now()).toMillis();

		System.out.println();
		ok("Import complete in " + ms + " ms");
		ok("  adapters imported : " + set.size());
		System.out.println();
		info("Checkpoint written to: " + out.toAbsolutePath());
		info("Use it with:  juno local --model-path model.gguf --lora-play " + out);
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static void printHelp() {
		System.out.println();
		System.out.println("  juno lora-import — convert a GGUF LoRA adapter into a Juno .lora v2 checkpoint");
		System.out.println();
		System.out.println("  Usage:");
		System.out.println("    juno lora-import --gguf adapter.gguf --out x.lora [--alpha N]");
		System.out.println();
		System.out.println("  Options:");
		System.out.println("    --gguf PATH    Source GGUF LoRA adapter (required)");
		System.out.println("    --out PATH     Destination .lora v2 checkpoint (required)");
		System.out.println("    --alpha N      Override alpha for every imported adapter (default: GGUF");
		System.out.println("                   \"adapter.lora.alpha\" metadata if present, else alpha == rank,");
		System.out.println("                   i.e. the decomposition is applied unscaled)");
		System.out.println("    --help         Show this message");
		System.out.println();
		System.out.println("  Expected tensor naming (converter reference, not embedded):");
		System.out.println("    blk.<layer>.<proj>.weight.lora_a / .lora_b, proj in attn_q/attn_k/attn_v/");
		System.out.println("    attn_output/ffn_gate/ffn_up/ffn_down. Unrecognized tensor names or a");
		System.out.println("    rank/shape mismatch between the two halves fail the import closed.");
		System.out.println();
		System.out.println("  Example workflow:");
		System.out.println("    juno lora-import --gguf hub-adapter.gguf --out hub-adapter.lora");
		System.out.println("    juno local --model-path model.gguf --lora-play hub-adapter.lora");
		System.out.println();
	}

	private static void info(String msg) {
		System.out.println("\033[0;36m▶ " + msg + "\033[0m");
	}

	private static void ok(String msg) {
		System.out.println("\033[0;32m✔ " + msg + "\033[0m");
	}

	private static void err(String msg) {
		System.err.println("\033[0;31m✖ " + msg + "\033[0m");
		System.exit(1);
	}
}
