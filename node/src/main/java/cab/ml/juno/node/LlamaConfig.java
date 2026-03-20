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

/**
 * LLaMA-family model configuration extracted from GGUF metadata.
 *
 * Supports LLaMA 2, LLaMA 3, Mistral, TinyLlama, and any model that uses the
 * standard llm.* or llama.* GGUF metadata keys.
 *
 * Usage: GgufReader reader = GgufReader.open(path); LlamaConfig cfg =
 * LlamaConfig.from(reader); System.out.println(cfg);
 */
public record LlamaConfig(int hiddenDim, // embedding / residual stream dimension
		int numLayers, // total transformer layers
		int numHeads, // number of query heads
		int numKvHeads, // number of KV heads (GQA — may differ from numHeads)
		int headDim, // dimension per head = hiddenDim / numHeads
		int intermediateSize, // FFN hidden dimension (SwiGLU gate/up projection width)
		int vocabSize, // vocabulary size
		float rmsNormEps, // RMS normalisation epsilon
		float ropeTheta, // RoPE base frequency
		String architecture // e.g. "llama"
) {

	/**
	 * Extract config from an open GgufReader. Reads standard GGUF metadata keys in
	 * priority order, falling back to llama.cpp legacy keys for older files.
	 *
	 * <h3>Vocab size — why we read from the tokenizer, not the architecture</h3>
	 * Some models (notably the Phi-3 family) store only the <em>base</em> vocabulary
	 * count in the architecture metadata key ({@code phi3.vocab_size = 32000}) while
	 * the actual tokenizer includes additional special tokens that push the true count
	 * higher ({@code tokenizer.ggml.tokens} array length = 32064 for phi-3.5-mini).
	 * The EOS token for phi-3.5-mini is ID 32000 — which sits exactly at the boundary
	 * of the arch vocab and is unreachable if we size the output projection to only
	 * 32000 rows.  We therefore use {@code tokenizer.ggml.tokens} array length as the
	 * authoritative vocab size whenever it is larger than the arch metadata value.
	 */
	public static LlamaConfig from(GgufReader r) {

		// Architecture prefix (llama, mistral, gemma, phi, …)
		String arch = r.metaString("general.architecture");
		if (arch == null)
			arch = "llama";
		String p = arch + ".";

		int hiddenDim = r.metaInt(p + "embedding_length", r.metaInt("llama.embedding_length", 2048));
		int numLayers = r.metaInt(p + "block_count", r.metaInt("llama.block_count", 22));
		int numHeads = r.metaInt(p + "attention.head_count", r.metaInt("llama.attention.head_count", 32));
		int numKvHeads = r.metaInt(p + "attention.head_count_kv", r.metaInt("llama.attention.head_count_kv", numHeads));

		// ── Vocab size: arch metadata vs. actual tokenizer count ──────────────
		// phi3.vocab_size (and equivalents for other arch prefixes) counts only the
		// base vocabulary.  tokenizer.ggml.tokens is the complete list including all
		// special tokens added by the model author.  We take the larger value so that
		// the output projection covers every token the tokenizer can produce.
		int archVocabSize = r.metaInt(p + "vocab_size", r.metaInt("llama.vocab_size", 32000));
		int tokenizerVocabSize = tokenizerTokenCount(r);
		int vocabSize = Math.max(archVocabSize, tokenizerVocabSize);

		int intermediateSize = r.metaInt(p + "feed_forward_length", r.metaInt("llama.feed_forward_length",
				// TinyLlama default 5632 (hidden 2048 * 2.75)
				hiddenDim * 11 / 4));
		float rmsNormEps = r.metaFloat(p + "attention.layer_norm_rms_epsilon",
				r.metaFloat("llama.attention.layer_norm_rms_epsilon", 1e-5f));
		float ropeTheta = r.metaFloat(p + "rope.freq_base", r.metaFloat("llama.rope.freq_base", 10000.0f));

		int headDim = hiddenDim / numHeads;

		return new LlamaConfig(hiddenDim, numLayers, numHeads, numKvHeads, headDim, intermediateSize, vocabSize,
				rmsNormEps, ropeTheta, arch);
	}

	/**
	 * Returns the number of tokens in {@code tokenizer.ggml.tokens}, or 0 if the
	 * key is absent.  This is the authoritative vocab count for models that add
	 * special tokens beyond their declared {@code arch.vocab_size}.
	 */
	private static int tokenizerTokenCount(GgufReader r) {
		Object v = r.meta("tokenizer.ggml.tokens");
		return (v instanceof Object[] arr) ? arr.length : 0;
	}

	/** Grouped-query attention ratio: how many Q-heads share each KV head. */
	public int gqaRatio() {
		return numHeads / numKvHeads;
	}

	/** KV dimension per head (same as headDim for standard models). */
	public int kvDim() {
		return numKvHeads * headDim;
	}

	@Override
	public String toString() {
		return String.format(
				"LlamaConfig{arch=%s hidden=%d layers=%d heads=%d kvHeads=%d headDim=%d"
						+ " ffn=%d vocab=%d eps=%.1e ropeTheta=%.0f}",
				architecture, hiddenDim, numLayers, numHeads, numKvHeads, headDim, intermediateSize, vocabSize,
				rmsNormEps, ropeTheta);
	}
}