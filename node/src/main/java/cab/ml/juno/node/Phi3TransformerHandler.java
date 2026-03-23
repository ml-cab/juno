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
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * CPU implementation of the Phi-3 family transformer forward pass.
 *
 * <h3>Phi-3 vs LLaMA tensor layout differences</h3>
 * <ol>
 *   <li><b>Fused QKV projection</b>: {@code blk.{i}.attn_qkv.weight} shape
 *       {@code [H + 2·kvDim, H]}.  LLaMA stores separate {@code attn_q/k/v}
 *       tensors.  This handler keeps the fused tensor in one
 *       {@link GgufReader.QuantizedTensor} and uses row-range matVec to extract
 *       Q (rows 0..H−1), K (rows H..H+kvDim−1), V (rows H+kvDim..end).
 *   <li><b>Fused gate+up FFN</b>: {@code blk.{i}.ffn_up.weight} shape
 *       {@code [2·intermediateSize, H]}.  Gate occupies rows 0..I−1, up rows
 *       I..2I−1.  Again kept fused and sliced at call-time.
 * </ol>
 *
 * <h3>Memory layout — why QuantizedTensor and not float[]</h3>
 * Previous versions called {@code GgufReader.tensor()} for every projection
 * weight, which dequantised the entire matrix to float32 eagerly:
 * <pre>
 *   phi-3.5-mini-instruct.Q4_K_M:
 *     32 layers × 7 projection matrices × avg ~65 MB (float32) ≈ 14.5 GB
 *     --heap 12g  →  OOM  →  Linux SIGKILL  ("Killed", no Java stack trace)
 * </pre>
 * All seven projection matrices are now stored as
 * {@link GgufReader.QuantizedTensor} (raw Q4_K bytes).
 * {@link LlamaTransformerHandler#matVec(GgufReader.QuantizedTensor, float[], int, int, int)}
 * dequantises one 256-element block at a time during the inner-product loop,
 * keeping the live float footprint at ≈1 kB instead of ≈65 MB per tensor.
 * <pre>
 *   Quantised weight memory (Q4_K, 4.5 bits/weight):
 *     32 layers × 7 matrices × avg ~9 MB (Q4_K raw) ≈ 2 GB  ≪  12 GB
 * </pre>
 * Small tensors (norm weights, token embeddings, output projection) are still
 * loaded as {@code float[]} because they are either tiny or require random
 * row-access patterns that are inconvenient with quantised storage.
 *
 * <h3>Thread safety</h3>
 * Each request uses an isolated KV-cache entry keyed by {@code requestId}.
 * Multiple threads may call {@link #forward} concurrently for distinct requests.
 */
public final class Phi3TransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Phi3TransformerHandler.class.getName());

	// ── Loaded weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	// Small tensors: dequantised to float[] (each is at most a few hundred MB)
	private final float[] tokenEmbd;   // [vocabSize × hiddenDim] – first node only
	private final float[] outputNorm;  // [hiddenDim]             – last node only
	private final float[] outputProj;  // [vocabSize × hiddenDim] – last node only

	private final float[][] attnNorm;  // [L][hiddenDim]
	private final float[][] ffnNorm;   // [L][hiddenDim]

	// Large tensors: kept in quantised form to avoid OOM.
	// Shapes (logical):
	//   attnQkv[li]  → [H + kvDim + kvDim,  H]   (fused Q, K, V projections)
	//   wo[li]       → [H,                  H]   (attention output projection)
	//   ffnGateUp[li]→ [2 × intermediateSize, H] (fused gate + up projections)
	//   wDown[li]    → [H,       intermediateSize]
	private final GgufReader.QuantizedTensor[] attnQkv;
	private final GgufReader.QuantizedTensor[] wo;
	private final GgufReader.QuantizedTensor[] ffnGateUp;
	private final GgufReader.QuantizedTensor[] wDown;

	// Per-request KV cache — lazily allocated and grown on demand.
	// Starts at INITIAL_SEQ_CAPACITY slots, doubles until MAX_SEQ_LEN.
	// Avoids the 554 MB eager pre-allocation that caused node JVM OOM during
	// multi-test runs against phi-3.5-mini with --heap 4g.
	private final Map<String, float[][]> kvCacheK = new HashMap<>();
	private final Map<String, float[][]> kvCacheV = new HashMap<>();
	private static final int MAX_SEQ_LEN        = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64; // grows on demand

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load weights from a Phi-3 GGUF file for the given shard range.
	 *
	 * @param modelPath path to the GGUF file (e.g. phi-3.5-mini-instruct.Q4_K_M.gguf)
	 * @param context   describes which layers/embeddings this node is responsible for
	 */
	public static Phi3TransformerHandler load(Path modelPath, ShardContext context) throws IOException {
		log.info("Loading Phi-3 GGUF shard: layers " + context.startLayer() + "–" + context.endLayer()
				+ "  embd=" + context.hasEmbeddings()
				+ "  outProj=" + context.hasOutputProjection()
				+ "  file=" + modelPath);

		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			log.info("Model: " + cfg);
			return new Phi3TransformerHandler(r, cfg, context);
		}
	}

	private Phi3TransformerHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx) throws IOException {
		this.cfg          = cfg;
		this.startLayer   = ctx.startLayer();
		this.endLayer     = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L    = endLayer - startLayer;
		int H    = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I    = cfg.intermediateSize();

		// ── Small tensors: dequantise eagerly (float[]) ───────────────────────
		// tokenEmbd: [vocabSize × H]. Required for embedding lookup (random row
		// access) — quantised storage would complicate the index calculation.
		// outputProj: [vocabSize × H]. Usually tied to tokenEmbd; either way it
		// is loaded once and the float[] kept for the output matVec.
		// Both fit in a few hundred MB for typical Phi-3 models.
		this.tokenEmbd  = hasEmbeddings  ? r.tensor("token_embd.weight")  : null;
		this.outputNorm = hasOutputProj  ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj  ? loadOutputProjection(r)        : null;

		attnNorm = new float[L][];
		ffnNorm  = new float[L][];

		// ── Large tensors: keep as QuantizedTensor (raw Q4_K bytes) ──────────
		attnQkv   = new GgufReader.QuantizedTensor[L];
		wo        = new GgufReader.QuantizedTensor[L];
		ffnGateUp = new GgufReader.QuantizedTensor[L];
		wDown     = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			log.fine("Loading Phi-3 layer " + i + " weights...");

			// Norm weights are F32 scalars (hiddenDim each) — tiny, keep as float[]
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li]  = r.tensor("blk." + i + ".ffn_norm.weight");

			// Projection tensors: load raw quantised bytes.
			// Logical shapes:
			//   attn_qkv.weight : [H + kvDim + kvDim, H]
			//   attn_output.weight : [H, H]
			//   ffn_up.weight   : [2*I, H]   (gate rows 0..I-1, up rows I..2I-1)
			//   ffn_down.weight : [H, I]
			attnQkv[li]   = r.tensorRaw("blk." + i + ".attn_qkv.weight");
			wo[li]        = r.tensorRaw("blk." + i + ".attn_output.weight");
			ffnGateUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li]     = r.tensorRaw("blk." + i + ".ffn_down.weight");

			logLayerMemory(i, H, kvDim, I, attnQkv[li], wo[li], ffnGateUp[li], wDown[li]);
		}

		log.info("Phi-3 shard loaded — " + L + " layers, "
				+ (hasEmbeddings ? "with embeddings, " : "")
				+ (hasOutputProj ? "with output projection" : "no output projection"));
	}

	private static void logLayerMemory(int layer, int H, int kvDim, int I,
			GgufReader.QuantizedTensor qkv, GgufReader.QuantizedTensor wo,
			GgufReader.QuantizedTensor gateUp, GgufReader.QuantizedTensor down) {
		long quantBytes = (long) qkv.data().length + wo.data().length
				+ gateUp.data().length + down.data().length;
		long float32Bytes = (long) (H + 2L * kvDim) * H * 4
				+ (long) H * H * 4
				+ 2L * I * H * 4
				+ (long) H * I * 4;
		log.fine(String.format(
				"Layer %d: quantised projection weights %.1f MB  (float32 equiv. would be %.1f MB)",
				layer, quantBytes / 1e6, float32Bytes / 1e6));
	}

	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight")) {
			return r.tensor("output.weight");
		}
		log.info("output.weight not found — using tied embeddings as output projection");
		return r.tensor("token_embd.weight");
	}

	// ── ForwardPassHandler ────────────────────────────────────────────────────

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long start = System.nanoTime();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			return ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
		}
	}

	@Override
	public boolean isReady() {
		return true;
	}

	// ── Transformer forward pass ──────────────────────────────────────────────

	private float[] getInitialActivation(ForwardRequest request) {
		if (hasEmbeddings) {
			int[] tokenIds = request.tokenIds();
			int tokenId = tokenIds[tokenIds.length - 1];
			// Clamp to actual embedding table size, not cfg.vocabSize() which may
			// be the arch-metadata base count (e.g. 32000 for phi3) rather than the
			// full tokenizer count (e.g. 32064 including special tokens).
			int actualVocab = tokenEmbd.length / cfg.hiddenDim();
			tokenId = Math.max(0, Math.min(tokenId, actualVocab - 1));
			float[] x = new float[cfg.hiddenDim()];
			System.arraycopy(tokenEmbd, tokenId * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
			return x;
		} else {
			float[] x = new float[request.activations().length];
			System.arraycopy(request.activations(), 0, x, 0, x.length);
			return x;
		}
	}

	private float[] runLayers(float[] x, String requestId, int pos) {
		int L    = endLayer - startLayer;
		int kvDim = cfg.kvDim();

		// Lazy initial allocation — 64 slots, grows on demand to avoid OOM.
		// phi-3.5-mini: eager 2048 slots = 554 MB per request; lazy = 17 MB initially.
		kvCacheK.computeIfAbsent(requestId, _ -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		kvCacheV.computeIfAbsent(requestId, _ -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);

		float[][] kCache = kvCacheK.get(requestId);
		float[][] vCache = kvCacheV.get(requestId);

		// Grow before writing at pos — transformerLayer writes kCacheLayer[pos * kvDim]
		ensureKvCapacity(kCache, pos, kvDim);
		ensureKvCapacity(vCache, pos, kvDim);

		for (int li = 0; li < L; li++)
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);
		return x;
	}

	/**
	 * Grows KV cache arrays to accommodate {@code pos}. Uses doubling growth,
	 * capped at {@link #MAX_SEQ_LEN}. Modifies the array slots in-place so
	 * the HashMap entry is updated automatically.
	 */
	private static void ensureKvCapacity(float[][] cache, int pos, int kvDim) {
		int required = (pos + 1) * kvDim;
		for (int li = 0; li < cache.length; li++) {
			if (cache[li].length < required) {
				int newLen = cache[li].length;
				while (newLen < required)
					newLen = Math.min(newLen * 2, MAX_SEQ_LEN * kvDim);
				cache[li] = java.util.Arrays.copyOf(cache[li], newLen);
			}
		}
	}

	/**
	 * Package-private for testing: number of sequence positions currently
	 * allocated in the K cache for the given request.
	 */
	int kvCacheAllocatedSlots(String requestId) {
		float[][] k = kvCacheK.get(requestId);
		return (k == null || k.length == 0) ? 0 : k[0].length / cfg.kvDim();
	}

	private float[] transformerLayer(float[] x, int li, int pos,
			float[] kCacheLayer, float[] vCacheLayer) {
		int H     = cfg.hiddenDim();
		int kvDim = cfg.kvDim();

		// ── Attention sub-layer ───────────────────────────────────────────────
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		// Q, K, V via row-range matVec on the fused QKV tensor.
		// attnQkv rows:  [0, H)        → Q  (H rows, output size H)
		//                [H, H+kvDim)  → K  (kvDim rows)
		//                [H+kvDim, -)  → V  (kvDim rows)
		float[] q = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, 0,          H,           H);
		float[] k = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, H,          H + kvDim,   H);
		float[] v = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, H + kvDim,  H + 2*kvDim, H);

		LlamaTransformerHandler.rope(q, pos, cfg.numHeads(),   cfg.headDim(), cfg.ropeTheta());
		LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut  = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, 0, H, H);
		float[] x2       = LlamaTransformerHandler.add(x, attnProj);

		// ── FFN sub-layer ─────────────────────────────────────────────────────
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/**
	 * SwiGLU: silu(gate(x)) * up(x) → down.
	 *
	 * Gate and up are fused in ffnGateUp; row ranges split them at call-time.
	 * ffnGateUp rows: [0, I)   → gate projection (SiLU input)
	 *                 [I, 2I)  → up   projection
	 */
	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = LlamaTransformerHandler.matVec(ffnGateUp[li], x, 0, I,     H);
		float[] up   = LlamaTransformerHandler.matVec(ffnGateUp[li], x, I, 2 * I, H);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		return LlamaTransformerHandler.matVec(wDown[li], hidden, 0, H, I);
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		// Use actual tensor dimensions, not cfg.vocabSize().  For phi3,
		// cfg.vocabSize() may be the arch-metadata base count (32000) while
		// outputProj.length encodes the full tokenizer vocab (32064), so using
		// cfg.vocabSize() would omit the special-token logits including EOS.
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xNorm, actualVocab, cfg.hiddenDim());
	}

	/**
	 * Grouped-query attention — identical logic to LlamaTransformerHandler.gqa().
	 */
	private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
		int H   = cfg.numHeads();
		int Hd  = cfg.headDim();
		int gqa = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] out    = new float[H * Hd];
		float[] scores = new float[seqLen];

		for (int h = 0; h < H; h++) {
			int kvHead = h / gqa;
			int qBase  = h * Hd;
			int kBase  = kvHead * Hd;

			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * cfg.kvDim() + kBase;
				for (int d = 0; d < Hd; d++)
					dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}

			LlamaTransformerHandler.softmax(scores, seqLen);

			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOffset = t * cfg.kvDim() + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++)
					out[outBase + d] += w * vCache[vOffset + d];
			}
		}
		return out;
	}
}