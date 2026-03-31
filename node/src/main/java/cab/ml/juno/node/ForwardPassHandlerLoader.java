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
import java.util.logging.Logger;

/**
 * Factory that selects the correct ForwardPassHandler implementation by reading
 * {@code general.architecture} from GGUF metadata.
 *
 * <p>
 * Architecture dispatch table:
 * <ul>
 * <li>{@code phi3} → {@link Phi3TransformerHandler}
 * <li>everything else → {@link LlamaTransformerHandler}
 * </ul>
 *
 * <p>
 * Adding support for a new architecture requires only:
 * <ol>
 * <li>Implement a new {@link ForwardPassHandler} subclass.
 * <li>Add a {@code case} branch here.
 * </ol>
 *
 * <p>
 * Usage:
 * 
 * <pre>{@code
 * ForwardPassHandler handler = ForwardPassHandlerLoader.load(modelPath, shardContext);
 * }</pre>
 */
public final class ForwardPassHandlerLoader {

	private static final Logger log = Logger.getLogger(ForwardPassHandlerLoader.class.getName());

	private ForwardPassHandlerLoader() {
	}

	/**
	 * Open the GGUF file, read {@code general.architecture}, and return the
	 * appropriate handler with weights fully loaded for the given shard.
	 *
	 * @param modelPath path to the GGUF file
	 * @param context   shard assignment (layers, embedding flags)
	 * @return a ready-to-use {@link ForwardPassHandler}
	 * @throws IOException if the file cannot be opened or a tensor is missing
	 */
	public static ForwardPassHandler load(Path modelPath, ShardContext context) throws IOException {
		String arch = readArchitecture(modelPath);
		log.info("Detected architecture: " + arch + "  file=" + modelPath);

		return switch (arch) {
		case "phi3" -> {
			log.info("Routing to Phi3TransformerHandler (phi3 fused-QKV architecture)");
			yield Phi3TransformerHandler.load(modelPath, context);
		}
		default -> {
			log.info("Routing to LlamaTransformerHandler (LLaMA-family architecture: " + arch + ")");
			yield LlamaTransformerHandler.load(modelPath, context);
		}
		};
	}

	/**
	 * Open the GGUF file, read {@code general.architecture}, and return the
	 * appropriate handler wired with the specified {@link MatVec}.
	 *
	 * <p>Uses {@link WeightDequantMode#EAGER} — weights are dequantized once and
	 * uploaded to the GPU when backend is {@link CudaMatVec}. Prefer
	 * {@link #load(Path, ShardContext, MatVec, WeightDequantMode)} to pass the mode
	 * explicitly.
	 *
	 * @param modelPath path to the GGUF file
	 * @param context   shard assignment (layers, embedding flags)
	 * @param backend   compute backend to inject into the handler
	 * @return a ready-to-use {@link ForwardPassHandler}
	 * @throws IOException if the file cannot be opened or a tensor is missing
	 */
	public static ForwardPassHandler load(Path modelPath, ShardContext context, MatVec backend)
			throws IOException {
		return load(modelPath, context, backend, WeightDequantMode.EAGER);
	}

	/**
	 * Open the GGUF file, read {@code general.architecture}, and return the
	 * appropriate handler wired with the specified {@link MatVec} and
	 * {@link WeightDequantMode}.
	 *
	 * <pre>{@code
	 * GpuContext ctx = GpuContext.init(0);
	 * ForwardPassHandler h = ForwardPassHandlerLoader.load(
	 *     modelPath, shard, new CudaMatVec(ctx), WeightDequantMode.EAGER);
	 * }</pre>
	 *
	 * @param modelPath path to the GGUF file
	 * @param context   shard assignment (layers, embedding flags)
	 * @param backend   compute backend to inject into the handler
	 * @param mode      {@link WeightDequantMode#EAGER} = dequantize once, upload to
	 *                  GPU; {@link WeightDequantMode#LAZY} = keep quantized, CPU
	 *                  dequantize per block on every decode step
	 * @return a ready-to-use {@link ForwardPassHandler}
	 * @throws IOException if the file cannot be opened or a tensor is missing
	 */
	public static ForwardPassHandler load(Path modelPath, ShardContext context, MatVec backend,
			WeightDequantMode mode) throws IOException {
		String arch = readArchitecture(modelPath);
		log.info("Detected architecture: " + arch + "  backend=" + backend.getClass().getSimpleName()
				+ "  dequant=" + mode.name().toLowerCase() + "  file=" + modelPath);

		return switch (arch) {
		case "phi3" -> {
			log.info("Routing to Phi3TransformerHandler (phi3 fused-QKV architecture)");
			// Phi3 uses its own CPU static matVec; dequant mode is not applicable.
			yield Phi3TransformerHandler.load(modelPath, context, backend);
		}
		default -> {
			log.info("Routing to LlamaTransformerHandler (LLaMA-family architecture: " + arch + ")");
			yield LlamaTransformerHandler.load(modelPath, context, backend, mode);
		}
		};
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	/**
	 * Opens the GGUF file just long enough to read the {@code general.architecture}
	 * metadata key, then closes it. The caller's full load opens the file again.
	 *
	 * @return architecture string, lower-cased; {@code "llama"} when absent
	 */
	private static String readArchitecture(Path modelPath) throws IOException {
		try (GgufReader r = GgufReader.open(modelPath)) {
			String arch = r.metaString("general.architecture");
			return arch != null ? arch.toLowerCase().strip() : "llama";
		}
	}
}