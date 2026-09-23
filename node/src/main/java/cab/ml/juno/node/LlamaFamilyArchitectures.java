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
import java.util.Set;

/**
 * The {@code general.architecture} values that {@link LlamaTransformerHandler}
 * is verified to run correctly.
 *
 * <p>
 * The handler implements one computation: pre-norm decoder blocks with RMS norm,
 * rotary embeddings, grouped-query attention and a SwiGLU feed-forward network.
 * An architecture outside this set can share tensor names with that layout while
 * computing something different (sliding-window attention, logit softcapping,
 * per-layer input embeddings, recurrent layers, routed experts). Loading such a
 * file into this handler either fails on an unrelated missing tensor or, worse,
 * finds every tensor it looks for and produces fluent-looking wrong output. So
 * the fallback is an allowlist of values that were checked against real model
 * files, and everything else is rejected by name.
 *
 * <p>
 * Architectures with their own handler ({@code phi2}, {@code phi3},
 * {@code qwen3}, {@code qwen3moe}) are dispatched before this check and are not
 * listed here.
 */
final class LlamaFamilyArchitectures {

	/**
	 * Dense Llama family (Llama, Mistral, TinyLlama, and derivatives that declare
	 * {@code llama}) and the Qwen2 family, which shares the layout.
	 */
	private static final Set<String> VERIFIED = Set.of("llama", "mistral", "tinyllama", "qwen2", "qwen2.5");

	private LlamaFamilyArchitectures() {
	}

	/** Whether {@code architecture} (already lower-cased and stripped) is in the verified set. */
	static boolean isVerified(String architecture) {
		return VERIFIED.contains(architecture);
	}

	/**
	 * @throws IOException naming the architecture when it is not verified
	 */
	static void requireVerified(String architecture, Path modelPath) throws IOException {
		if (isVerified(architecture))
			return;
		throw new IOException("Unsupported model architecture '" + architecture + "' in " + modelPath
				+ ". Supported: llama, mistral, tinyllama, qwen2 (dense Llama family), phi2, phi3, qwen3, qwen3moe. "
				+ "This architecture computes something the dense Llama path does not (for example sliding-window "
				+ "attention, logit softcapping, recurrent layers or routed experts), so loading it would produce "
				+ "wrong output.");
	}
}
