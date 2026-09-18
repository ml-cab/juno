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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Imports a GGUF LoRA adapter (as produced by the common
 * {@code convert_lora_to_gguf.py}-style converter, referenced for tensor
 * naming/layout only — this reader does not depend on or embed that tool)
 * into a Juno {@code .lora} v2 {@link LoraAdapterSet}.
 *
 * <p>
 * Expected tensor naming: {@code blk.<layer>.<ggml_proj>.weight.lora_a} and
 * {@code blk.<layer>.<ggml_proj>.weight.lora_b} (an optional trailing
 * {@code .weight} on the lora suffix is also accepted, since converter
 * versions vary). {@code ggml_proj} is one of the standard GGML tensor-name
 * projections ({@code attn_q}, {@code attn_k}, {@code attn_v},
 * {@code attn_output}, {@code ffn_gate}, {@code ffn_up}, {@code ffn_down}),
 * mapped to Juno's own projection keys ({@code wq}/{@code wk}/{@code wv}/
 * {@code wo}/{@code wgate}/{@code wup}/{@code wdown}).
 *
 * <p>
 * Tensor layout: GGUF stores dims fastest-varying first ({@code ne[0]}
 * innermost), so a {@code lora_a} tensor of shape ({@code rank}, {@code inDim})
 * is expected with {@code ne = [inDim, rank]}, and {@code lora_b}
 * ({@code outDim}, {@code rank}) with {@code ne = [rank, outDim]} — both
 * already match {@link LoraAdapter}'s row-major {@code A}/{@code B} storage
 * once read via {@link GgufReader#tensor(String)}, so no transpose is needed.
 *
 * <p>
 * <b>Not verified against a real converter-produced GGUF-LoRA file this
 * session</b> (no network access to fetch one) — the naming/layout
 * convention above follows the commonly documented scheme; a mismatched real
 * file will fail closed with a clear error (unrecognized tensor name, rank
 * mismatch, or non-2-D tensor) rather than silently importing wrong weights.
 *
 * <p>
 * Every tensor in the file must be a recognized {@code lora_a}/{@code lora_b}
 * half; an unrecognized tensor name fails the whole import closed rather
 * than being silently skipped, since a skipped tensor would silently produce
 * an incomplete adapter.
 */
public final class GgufLoraImporter {

	private static final Map<String, String> PROJ_ALIASES = Map.of(
			"attn_q", "wq",
			"attn_k", "wk",
			"attn_v", "wv",
			"attn_output", "wo",
			"ffn_gate", "wgate",
			"ffn_up", "wup",
			"ffn_down", "wdown");

	private static final Pattern TENSOR_NAME = Pattern.compile("^blk\\.(\\d+)\\.([a-z_]+)\\.weight\\.lora_(a|b)(?:\\.weight)?$");

	private static final String ALPHA_META_KEY = "adapter.lora.alpha";

	private GgufLoraImporter() {
	}

	/**
	 * @param alphaOverride when present, used as the declared alpha for every
	 *                      imported adapter, overriding GGUF metadata. When
	 *                      absent, falls back to the {@value #ALPHA_META_KEY}
	 *                      GGUF metadata float if present, else to
	 *                      {@code alpha == rank} (scale 1.0 — apply the
	 *                      decomposition unscaled, the neutral default when no
	 *                      alpha is declared anywhere).
	 */
	public static LoraAdapterSet importFrom(Path gguf, OptionalDouble alphaOverride) throws IOException {
		try (GgufReader r = GgufReader.open(gguf)) {
			return importFrom(r, alphaOverride);
		}
	}

	static LoraAdapterSet importFrom(GgufReader r, OptionalDouble alphaOverride) throws IOException {
		Map<String, TensorPair> pairs = new LinkedHashMap<>();
		for (String name : r.tensorNames()) {
			Matcher m = TENSOR_NAME.matcher(name);
			if (!m.matches())
				throw new IllegalArgumentException("Unrecognized tensor in GGUF LoRA adapter: " + name
						+ "  (expected blk.<layer>.<proj>.weight.lora_a/lora_b)");
			int layer = Integer.parseInt(m.group(1));
			String ggmlProj = m.group(2);
			String proj = PROJ_ALIASES.get(ggmlProj);
			if (proj == null)
				throw new IllegalArgumentException("Unsupported LoRA projection '" + ggmlProj + "' in tensor " + name
						+ "  (known: " + PROJ_ALIASES.keySet() + ")");
			boolean isA = m.group(3).equals("a");
			TensorPair pair = pairs.computeIfAbsent(layer + ":" + proj, k -> new TensorPair());
			if (isA)
				pair.aName = name;
			else
				pair.bName = name;
		}
		if (pairs.isEmpty())
			throw new IllegalArgumentException(
					"No LoRA tensors found — expected names like blk.0.attn_q.weight.lora_a/lora_b");

		float metaAlpha = r.metaFloat(ALPHA_META_KEY, Float.NaN);

		LoraAdapterSet set = new LoraAdapterSet();
		for (Map.Entry<String, TensorPair> e : pairs.entrySet()) {
			String key = e.getKey();
			TensorPair pair = e.getValue();
			if (pair.aName == null)
				throw new IllegalArgumentException("LoRA tensor group " + key + " is missing its lora_a half");
			if (pair.bName == null)
				throw new IllegalArgumentException("LoRA tensor group " + key + " is missing its lora_b half");

			long[] aDims = r.tensorDims(pair.aName);
			long[] bDims = r.tensorDims(pair.bName);
			if (aDims.length != 2 || bDims.length != 2)
				throw new IllegalArgumentException("LoRA tensor group " + key + " must be 2-D; got lora_a="
						+ Arrays.toString(aDims) + " lora_b=" + Arrays.toString(bDims));

			int inDim = Math.toIntExact(aDims[0]);
			int rankA = Math.toIntExact(aDims[1]);
			int rankB = Math.toIntExact(bDims[0]);
			int outDim = Math.toIntExact(bDims[1]);
			if (rankA != rankB)
				throw new IllegalArgumentException(
						"LoRA tensor group " + key + " rank mismatch: lora_a rank=" + rankA + " vs lora_b rank=" + rankB);
			if (inDim < 1 || outDim < 1 || rankA < 1)
				throw new IllegalArgumentException(
						"LoRA tensor group " + key + " has an invalid dimension: in=" + inDim + " out=" + outDim
								+ " rank=" + rankA);

			float[] a = r.tensor(pair.aName);
			float[] b = r.tensor(pair.bName);

			float alpha = alphaOverride.isPresent() ? (float) alphaOverride.getAsDouble()
					: !Float.isNaN(metaAlpha) ? metaAlpha : (float) rankA;

			LoraAdapter adapter = LoraAdapter.fromWeights(LoraAdapterConfig.of(rankA, alpha), inDim, outDim, a, b);
			int layer = LoraAdapterSet.keyLayer(key);
			String proj = LoraAdapterSet.keyProj(key);
			set.add(layer, proj, adapter);
		}
		return set;
	}

	private static final class TensorPair {
		String aName;
		String bName;
	}
}
