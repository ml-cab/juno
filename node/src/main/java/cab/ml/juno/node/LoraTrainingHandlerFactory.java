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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Selects a {@link LoraTrainingHandler} by GGUF {@code general.architecture}.
 *
 * <p>Explicit allowlist (Tier 6):
 * <ul>
 * <li>{@code llama}, {@code mistral}, {@code tinyllama} → {@link LoraTrainableHandler}
 * <li>{@code qwen2}, {@code qwen2.5} → {@link Qwen2LoraTrainableHandler}
 * <li>{@code phi3} → {@link Phi3LoraTrainableHandler}
 * <li>{@code qwen3} (dense) → {@link Qwen3LoraTrainableHandler}
 * <li>{@code qwen3moe}, {@code qwen35}, {@code gemma}, unknown → rejected
 * </ul>
 */
public final class LoraTrainingHandlerFactory {

	private static final Logger log = Logger.getLogger(LoraTrainingHandlerFactory.class.getName());

	private static final Set<String> LLAMA_FAMILY = Set.of("llama", "mistral", "tinyllama");
	private static final Set<String> QWEN2_FAMILY = Set.of("qwen2", "qwen2.5");

	private LoraTrainingHandlerFactory() {
	}

	/**
	 * Whether LoRA training/playback is supported for this architecture string.
	 */
	public static boolean isSupported(String architecture) {
		String a = normalize(architecture);
		return LLAMA_FAMILY.contains(a) || QWEN2_FAMILY.contains(a) || "phi3".equals(a) || "qwen3".equals(a);
	}

	/**
	 * Throws if architecture is not on the Tier 6 LoRA allowlist.
	 */
	public static void requireSupported(String architecture) {
		String a = normalize(architecture);
		if (isSupported(a))
			return;
		throw new IllegalArgumentException("LoRA is not supported for architecture '" + a
				+ "'; supported: llama, mistral, tinyllama, qwen2, qwen2.5, phi3, qwen3 (dense). "
				+ "Rejected: qwen3moe, qwen35, gemma, and unknown architectures.");
	}

	public static LoraTrainingHandler create(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		return create(modelPath, context, adapters, ForwardPassHandlerLoader.selectLoraBackend());
	}

	public static LoraTrainingHandler create(Path modelPath, ShardContext context, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		String arch = readArchitecture(modelPath);
		requireSupported(arch);
		log.info("LoRA factory: arch=" + arch + " backend=" + backend.getClass().getSimpleName() + " file="
				+ modelPath);
		String a = normalize(arch);
		if (LLAMA_FAMILY.contains(a))
			return LoraTrainableHandler.load(modelPath, context, adapters, backend);
		if (QWEN2_FAMILY.contains(a))
			return Qwen2LoraTrainableHandler.load(modelPath, context, adapters, backend);
		if ("phi3".equals(a))
			return Phi3LoraTrainableHandler.load(modelPath, context, adapters, backend);
		if ("qwen3".equals(a))
			return Qwen3LoraTrainableHandler.load(modelPath, context, adapters, backend);
		throw new IllegalArgumentException("LoRA is not supported for architecture '" + a + "'");
	}

	static String normalize(String architecture) {
		if (architecture == null || architecture.isBlank())
			return "llama";
		return architecture.toLowerCase(Locale.ROOT).strip();
	}

	static String readArchitecture(Path modelPath) throws IOException {
		try (GgufReader r = GgufReader.open(modelPath)) {
			String arch = r.metaString("general.architecture");
			return arch != null ? arch.toLowerCase(Locale.ROOT).strip() : "llama";
		}
	}
}
