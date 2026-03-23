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
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * LLaMA-family transformer handler with LoRA fine-tuning support.
 *
 * <p>
 * Implements both inference and a complete training step. Frozen base weights
 * are kept quantised (Q4_K, Q8_0, etc.) for inference. During a training step
 * the same quantised weights are also used for the <em>backward</em> transpose
 * matmul — one row at a time, dequantised on demand, so peak extra memory is
 * O(hiddenDim) not O(model).
 *
 * <h2>LoRA configuration</h2> Create a {@link LoraAdapterSet} and pass it to
 * {@link #load}. Adapters are typically applied to {@code wq} and {@code wv}.
 * Any projection with no registered adapter is left frozen and does not
 * generate parameter gradients.
 *
 * <h2>Training loop</h2>
 * 
 * <pre>
 * LoraAdapterSet adapters = LoraAdapterSet.qv(cfg, rank = 8, alpha = 8f, rng);
 * LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
 * LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-4);
 *
 * for (String doc : docs) {
 * 	int[] tokens = tokenize(doc);
 * 	adapters.zeroAllGrads();
 * 	float loss = handler.trainStep(tokens, opt);
 * 	System.out.printf("loss=%.4f%n", loss);
 * }
 * adapters.save(Path.of("checkpoint.lora"));
 * </pre>
 *
 * <h2>Truncated BPTT</h2> Gradients do NOT propagate through the KV-cache
 * entries from earlier sequence positions. This is the standard simplification
 * for LoRA training: each position's backward is independent, so the total
 * gradient is the sum over positions. In practice this has negligible impact on
 * quality for typical fine-tuning sequences (≤ 512 tokens).
 *
 * <h2>Thread safety</h2> {@link #forward} is safe to call concurrently for
 * distinct request IDs. {@link #trainStep} is NOT thread-safe — it mutates
 * gradient accumulators.
 */
public final class LoraTrainableHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(LoraTrainableHandler.class.getName());

	// ── Activations stored per layer during a training forward ────────────────

	private record LayerState(float[] xIn, // residual stream before this layer [H]
			float[] xNorm1, // after pre-attention rmsNorm [H]
			float[] qPreRope, // Q projection before RoPE [numHeads*headDim]
			float[][] attnW, // attention weights per head [numHeads][seqLen]
			float[] xRes2, // after attention residual (= xIn + attnProj) [H]
			float[] xNorm2, // after pre-FFN rmsNorm [H]
			float[] gate, // FFN gate output [I]
			float[] up, // FFN up output [I]
			float[] hiddenAct // silu(gate) * up [I]
	) {
	}

	// ── Frozen weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final int startLayer, endLayer;
	private final boolean hasEmbeddings, hasOutputProj;

	private final float[] tokenEmbd; // [vocabSize × hiddenDim] or null
	private final float[] outputNorm; // [hiddenDim] or null
	private final GgufReader.QuantizedTensor outputProj; // or null

	private final float[][] attnNorm; // [L][hiddenDim]
	private final float[][] ffnNorm; // [L][hiddenDim]
	private final GgufReader.QuantizedTensor[] wq, wk, wv, wo;
	private final GgufReader.QuantizedTensor[] wGate, wUp, wDown;

	// ── LoRA adapters ─────────────────────────────────────────────────────────

	private final LoraAdapterSet loraAdapters;

	// ── Inference KV cache ────────────────────────────────────────────────────

	private final Map<String, float[][]> kvCacheK = new HashMap<>();
	private final Map<String, float[][]> kvCacheV = new HashMap<>();
	private static final int MAX_SEQ_LEN = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64;

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load a model shard and attach LoRA adapters.
	 *
	 * @param modelPath path to the GGUF file
	 * @param context   which layers/embeddings this node is responsible for
	 * @param adapters  LoRA adapters (typically created with
	 *                  {@link LoraAdapterSet#qv})
	 */
	public static LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		log.info("Loading LoRA handler: layers " + context.startLayer() + "–" + context.endLayer() + "  adapters="
				+ adapters.size() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new LoraTrainableHandler(r, cfg, context, adapters);
		}
	}

	private LoraTrainableHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, LoraAdapterSet adapters)
			throws IOException {
		this.cfg = cfg;
		this.loraAdapters = adapters;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L = endLayer - startLayer;

		this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

		attnNorm = new float[L][];
		ffnNorm = new float[L][];
		wq = new GgufReader.QuantizedTensor[L];
		wk = new GgufReader.QuantizedTensor[L];
		wv = new GgufReader.QuantizedTensor[L];
		wo = new GgufReader.QuantizedTensor[L];
		wGate = new GgufReader.QuantizedTensor[L];
		wUp = new GgufReader.QuantizedTensor[L];
		wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");
			wq[li] = r.tensorRaw("blk." + i + ".attn_q.weight");
			wk[li] = r.tensorRaw("blk." + i + ".attn_k.weight");
			wv[li] = r.tensorRaw("blk." + i + ".attn_v.weight");
			wo[li] = r.tensorRaw("blk." + i + ".attn_output.weight");
			wGate[li] = r.tensorRaw("blk." + i + ".ffn_gate.weight");
			wUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");
		}
	}

	private static GgufReader.QuantizedTensor loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight"))
			return r.tensorRaw("output.weight");
		log.info("output.weight absent — using tied embeddings");
		return r.tensorRaw("token_embd.weight");
	}

	// ── ForwardPassHandler (inference) ────────────────────────────────────────

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long t0 = System.nanoTime();
		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());
		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - t0);
		}
		return ForwardResult.activations(request.requestId(), x, System.nanoTime() - t0);
	}

	@Override
	public boolean isReady() {
		return true;
	}

	// ── Training step ─────────────────────────────────────────────────────────

	/**
	 * One teacher-forcing training step over a token sequence.
	 *
	 * <p>
	 * For tokens [t₀, t₁, …, t_{n}], position {@code pos} predicts
	 * {@code tokens[pos+1]}. The loss is the mean cross-entropy across all
	 * positions. Gradients accumulate into each {@link LoraAdapter}'s
	 * {@code gradA}/{@code gradB}. The optimizer step is applied inside this
	 * method; call {@link LoraAdapterSet#zeroAllGrads()} before the next step.
	 *
	 * @param tokens    input token sequence, length ≥ 2
	 * @param optimizer Adam optimizer to apply after backward
	 * @return mean cross-entropy loss (nats) for this sequence
	 */
	public float trainStep(int[] tokens, LoraAdamOptimizer optimizer) {
		if (tokens.length < 2)
			throw new IllegalArgumentException("tokens.length must be >= 2 (need at least one prediction pair)");

		int T = tokens.length - 1;
		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();

		LayerState[][] allStates = new LayerState[T][L];
		float[][] allXFinal = new float[T][];
		float[][] allXNormFinal = new float[T][];
		float[][] allProbs = new float[T][];

		float[][] kCache = new float[L][T * kvDim];
		float[][] vCache = new float[L][T * kvDim];

		LoraTrainEvent event = new LoraTrainEvent();
		event.begin();
		event.numTokens = tokens.length;
		event.step = optimizer.step() + 1; // will be the step number after this call

		// ── Forward ───────────────────────────────────────────────────────────
		long t0 = System.currentTimeMillis();
		for (int pos = 0; pos < T; pos++) {
			float[] x = embedding(tokens[pos]);
			for (int li = 0; li < L; li++) {
				allStates[pos][li] = forwardLayerStore(x, li, pos, kCache[li], vCache[li]);
				x = computeLayerOutput(allStates[pos][li], li, x);
			}
			if (hasOutputProj) {
				allXFinal[pos] = x.clone();
				allXNormFinal[pos] = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
				float[] logits = LlamaTransformerHandler.matVec(outputProj, allXNormFinal[pos], cfg.vocabSize(), H);
				allProbs[pos] = softmaxCopy(logits);
			}
		}
		event.forwardMs = System.currentTimeMillis() - t0;

		// ── Loss ──────────────────────────────────────────────────────────────
		float loss = 0f;
		for (int pos = 0; pos < T; pos++) {
			int target = tokens[pos + 1];
			if (allProbs[pos] != null)
				loss -= (float) Math.log(Math.max(allProbs[pos][target], 1e-9f));
		}
		loss /= T;

		// ── Backward ──────────────────────────────────────────────────────────
		long t1 = System.currentTimeMillis();
		loraAdapters.zeroAllGrads();

		for (int pos = 0; pos < T; pos++) {
			int target = tokens[pos + 1];

			float[] gradX;
			if (hasOutputProj) {
				float[] gradLogits = allProbs[pos].clone();
				gradLogits[target] -= 1.0f;
				for (int i = 0; i < gradLogits.length; i++)
					gradLogits[i] /= T;

				float[] gradXNormFinal = transposedMatVec(outputProj, gradLogits, cfg.vocabSize(), H);
				gradX = rmsNormBackward(allXFinal[pos], outputNorm, gradXNormFinal);
			} else {
				gradX = new float[H];
			}

			for (int li = L - 1; li >= 0; li--) {
				gradX = backwardLayer(gradX, li, pos, allStates[pos][li], kCache[li], vCache[li], pos + 1);
			}
		}
		event.backwardMs = System.currentTimeMillis() - t1;

		// ── Optimizer step ────────────────────────────────────────────────────
		long t2 = System.currentTimeMillis();
		optimizer.step(loraAdapters);
		event.optimizerMs = System.currentTimeMillis() - t2;

		event.loss = loss;
		event.totalMs = event.forwardMs + event.backwardMs + event.optimizerMs;
		event.commit();

		return loss;
	}

	// ── Inference helpers ─────────────────────────────────────────────────────

	private float[] getInitialActivation(ForwardRequest req) {
		if (hasEmbeddings) {
			int[] ids = req.tokenIds();
			int id = Math.max(0, Math.min(ids[ids.length - 1], cfg.vocabSize() - 1));
			float[] x = new float[cfg.hiddenDim()];
			System.arraycopy(tokenEmbd, id * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
			return x;
		}
		return req.activations().clone();
	}

	private float[] runLayers(float[] x, String requestId, int pos) {
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();
		kvCacheK.computeIfAbsent(requestId, _ -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		kvCacheV.computeIfAbsent(requestId, _ -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		float[][] kC = kvCacheK.get(requestId);
		float[][] vC = kvCacheV.get(requestId);
		ensureKvCapacity(kC, pos, kvDim);
		ensureKvCapacity(vC, pos, kvDim);
		for (int li = 0; li < L; li++) {
			x = inferenceLayer(x, li, pos, kC[li], vC[li]);
		}
		return x;
	}

	/** Fast inference layer — LoRA applied but no activations stored. */
	private float[] inferenceLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[] q = LlamaTransformerHandler.matVec(wq[li], xNorm1, H, H);
		float[] k = LlamaTransformerHandler.matVec(wk[li], xNorm1, kvDim, H);
		float[] v = LlamaTransformerHandler.matVec(wv[li], xNorm1, kvDim, H);

		// Apply LoRA deltas
		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		LlamaTransformerHandler.rope(q, pos, cfg.numHeads(), cfg.headDim(), cfg.ropeTheta());
		LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, H, H);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	private void applyLoraInPlace(float[] out, int li, String proj, float[] input) {
		LoraAdapter lora = loraAdapters.get(li + startLayer, proj);
		if (lora == null)
			return;
		float[] delta = lora.forward(input);
		for (int i = 0; i < out.length; i++)
			out[i] += delta[i];
	}

	private float[] outputProjection(float[] x) {
		float[] xn = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		return LlamaTransformerHandler.matVec(outputProj, xn, cfg.vocabSize(), cfg.hiddenDim());
	}

	// ── Training forward (with state capture) ─────────────────────────────────

	private float[] embedding(int tokenId) {
		if (!hasEmbeddings)
			throw new IllegalStateException("This shard does not own embeddings");
		tokenId = Math.max(0, Math.min(tokenId, cfg.vocabSize() - 1));
		float[] x = new float[cfg.hiddenDim()];
		System.arraycopy(tokenEmbd, tokenId * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
		return x;
	}

	/**
	 * Forward pass through one layer, storing all intermediate activations needed
	 * for the backward pass. Returns a {@link LayerState}; the caller must also
	 * call {@link #computeLayerOutput} to get the updated {@code x}.
	 */
	private LayerState forwardLayerStore(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int NH = cfg.numHeads();
		int Hd = cfg.headDim();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[] q = LlamaTransformerHandler.matVec(wq[li], xNorm1, H, H);
		float[] k = LlamaTransformerHandler.matVec(wk[li], xNorm1, kvDim, H);
		float[] v = LlamaTransformerHandler.matVec(wv[li], xNorm1, kvDim, H);

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		float[] qPreRope = q.clone(); // saved before RoPE for backward

		LlamaTransformerHandler.rope(q, pos, cfg.numHeads(), Hd, cfg.ropeTheta());
		LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), Hd, cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		// Attention — also capture per-head weights for backward
		int seqLen = pos + 1;
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] attnOut = new float[H];
		float[][] attnW = new float[NH][seqLen];
		float[] scores = new float[seqLen];

		int gqaR = cfg.gqaRatio();
		for (int h = 0; h < NH; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;
			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOff = t * kvDim + kBase;
				for (int d = 0; d < Hd; d++)
					dot += q[qBase + d] * kCacheLayer[kOff + d];
				scores[t] = dot * scale;
			}
			// softmax in-place on scores[0..seqLen)
			float max = Float.NEGATIVE_INFINITY;
			for (int t = 0; t < seqLen; t++)
				if (scores[t] > max)
					max = scores[t];
			float sum = 0f;
			for (int t = 0; t < seqLen; t++) {
				scores[t] = (float) Math.exp(scores[t] - max);
				sum += scores[t];
			}
			for (int t = 0; t < seqLen; t++) {
				scores[t] /= sum;
				attnW[h][t] = scores[t];
			}

			// weighted sum of values
			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOff = t * kvDim + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++)
					attnOut[outBase + d] += w * vCacheLayer[vOff + d];
			}
		}

		float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, H, H);
		float[] xRes2 = LlamaTransformerHandler.add(x, attnProj);
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(xRes2, ffnNorm[li], cfg.rmsNormEps());

		int I = cfg.intermediateSize();
		float[] gate = LlamaTransformerHandler.matVec(wGate[li], xNorm2, I, H);
		float[] up = LlamaTransformerHandler.matVec(wUp[li], xNorm2, I, H);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];

		return new LayerState(x.clone(), xNorm1, qPreRope, attnW, xRes2, xNorm2, gate, up, hidden);
	}

	/** Compute the layer output from stored state (completes forwardLayerStore). */
	private float[] computeLayerOutput(LayerState st, int li, float[] xIn) {
		int H = cfg.hiddenDim();
		float[] ffnOut = LlamaTransformerHandler.matVec(wDown[li], st.hiddenAct(), H, cfg.intermediateSize());
		return LlamaTransformerHandler.add(st.xRes2(), ffnOut);
	}

	// ── Backward ──────────────────────────────────────────────────────────────

	/**
	 * Backpropagate through one transformer layer. Accumulates gradients into LoRA
	 * adapters and returns dL/dx for the previous layer. Uses truncated BPTT: no
	 * gradients flow backward through the KV cache entries from earlier positions.
	 */
	private float[] backwardLayer(float[] gradOut, int li, int pos, LayerState st, float[] kCacheLayer,
			float[] vCacheLayer, int seqLen) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		int kvDim = cfg.kvDim();
		int NH = cfg.numHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();

		// ── FFN residual: xOut = xRes2 + ffnOut ──────────────────────────────
		float[] gradXRes2 = gradOut.clone(); // through residual
		float[] gradFfnOut = gradOut; // to FFN

		// Backward through wDown: ffnOut = wDown × hiddenAct
		float[] gradHidden = transposedMatVec(wDown[li], gradFfnOut, H, I);

		// Backward through SwiGLU: hiddenAct[i] = silu(gate[i]) * up[i]
		float[] gradGate = new float[I];
		float[] gradUp = new float[I];
		for (int i = 0; i < I; i++) {
			float g = st.gate()[i];
			float sig = 1f / (1f + (float) Math.exp(-g));
			gradGate[i] = gradHidden[i] * st.up()[i] * sig * (1f + g * (1f - sig));
			gradUp[i] = gradHidden[i] * LlamaTransformerHandler.silu(g);
		}

		// Backward through wGate and wUp
		float[] gradXNorm2 = add(transposedMatVec(wGate[li], gradGate, I, H), transposedMatVec(wUp[li], gradUp, I, H));

		// Backward through rmsNorm2
		addInPlace(gradXRes2, rmsNormBackward(st.xRes2(), ffnNorm[li], gradXNorm2));

		// ── Attention residual: xRes2 = xIn + attnProj ───────────────────────
		float[] gradXIn = gradXRes2.clone(); // through residual
		float[] gradAttnProj = gradXRes2;

		// Backward through wo: attnProj = wo × attnOut
		float[] gradAttnOut = transposedMatVec(wo[li], gradAttnProj, H, H);

		// ── Attention backward ────────────────────────────────────────────────
		// For each head: backprop through softmax + value weighted-sum
		// Output: gradQ (for LoRA Q), gradV at current pos (for LoRA V)
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] gradQ = new float[NH * Hd]; // Q gradient (pre-RoPE)
		float[] gradV = new float[kvDim]; // V gradient for current position only

		for (int h = 0; h < NH; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;
			float[] aw = st.attnW()[h]; // [seqLen]

			float[] gradAttnOut_h = Arrays.copyOfRange(gradAttnOut, qBase, qBase + Hd);

			// 1. gradV at current position (truncated BPTT: only pos contribution)
			for (int d = 0; d < Hd; d++)
				gradV[kBase + d] += aw[pos] * gradAttnOut_h[d];

			// 2. Softmax backward through attention weights
			// dL/dattnW[t] = dot(gradAttnOut_h, V[t])
			float[] dotWithV = new float[seqLen];
			for (int t = 0; t < seqLen; t++) {
				int vOff = t * kvDim + kBase;
				float d2 = 0f;
				for (int d = 0; d < Hd; d++)
					d2 += gradAttnOut_h[d] * vCacheLayer[vOff + d];
				dotWithV[t] = d2;
			}
			// Σ_t aw[t] * dotWithV[t]
			float sumDot = 0f;
			for (int t = 0; t < seqLen; t++)
				sumDot += aw[t] * dotWithV[t];
			// gradScores[t] = aw[t] * (dotWithV[t] - sumDot)
			float[] gradScores = new float[seqLen];
			for (int t = 0; t < seqLen; t++)
				gradScores[t] = aw[t] * (dotWithV[t] - sumDot);

			// 3. gradQ[h] = scale * Σ_t gradScores[t] * K[t]
			for (int t = 0; t < seqLen; t++) {
				if (gradScores[t] == 0f)
					continue;
				int kOff = t * kvDim + kBase;
				float gs = gradScores[t] * scale;
				for (int d = 0; d < Hd; d++)
					gradQ[qBase + d] += gs * kCacheLayer[kOff + d];
			}
		}

		// RoPE backward on gradQ (inverse rotation = R(-angle))
		ropeBackward(gradQ, pos, NH, Hd, cfg.ropeTheta());

		// ── LoRA / frozen projection backward ─────────────────────────────────
		// wq: xNorm1 → q. q = frozen_wq * xNorm1 + lora_q(xNorm1)
		float[] gradXNorm1 = transposedMatVec(wq[li], gradQ, H, H);
		LoraAdapter loraQ = loraAdapters.get(li + startLayer, "wq");
		if (loraQ != null)
			addInPlace(gradXNorm1, loraQ.backward(gradQ, st.xNorm1()));

		// wv: xNorm1 → v. v = frozen_wv * xNorm1 + lora_v(xNorm1)
		float[] gradXNorm1_v = transposedMatVec(wv[li], gradV, kvDim, H);
		LoraAdapter loraV = loraAdapters.get(li + startLayer, "wv");
		if (loraV != null)
			addInPlace(gradXNorm1_v, loraV.backward(gradV, st.xNorm1()));
		addInPlace(gradXNorm1, gradXNorm1_v);

		// Backward through rmsNorm1
		addInPlace(gradXIn, rmsNormBackward(st.xIn(), attnNorm[li], gradXNorm1));

		return gradXIn;
	}

	// ── Math helpers ──────────────────────────────────────────────────────────

	/**
	 * In-place softmax over a slice. Returns a new float[] (doesn't mutate input).
	 */
	private static float[] softmaxCopy(float[] logits) {
		float[] out = logits.clone();
		LlamaTransformerHandler.softmax(out, out.length);
		return out;
	}

	/**
	 * Grouped-query attention (inference, same as LlamaTransformerHandler).
	 */
	private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
		int H = cfg.numHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] out = new float[H * Hd];
		float[] scores = new float[seqLen];
		for (int h = 0; h < H; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;
			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOff = t * cfg.kvDim() + kBase;
				for (int d = 0; d < Hd; d++)
					dot += q[qBase + d] * kCache[kOff + d];
				scores[t] = dot * scale;
			}
			LlamaTransformerHandler.softmax(scores, seqLen);
			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOff = t * cfg.kvDim() + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++)
					out[outBase + d] += w * vCache[vOff + d];
			}
		}
		return out;
	}

	/** SwiGLU FFN (inference). */
	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = LlamaTransformerHandler.matVec(wGate[li], x, I, H);
		float[] up = LlamaTransformerHandler.matVec(wUp[li], x, I, H);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		return LlamaTransformerHandler.matVec(wDown[li], hidden, H, I);
	}

	/**
	 * RMSNorm backward.
	 * 
	 * <pre>
	 *   y_i = w_i * x_i * scale,  scale = 1/sqrt(mean(x^2) + eps)
	 *   dL/dx_j = w_j * scale * gradOut_j
	 *           - x_j * (scale^3 / n) * sum_i(gradOut_i * w_i * x_i)
	 * </pre>
	 */
	static float[] rmsNormBackward(float[] x, float[] w, float[] gradOut) {
		int n = x.length;
		float eps = 1e-5f;
		float ss = 0f;
		for (float v : x)
			ss += v * v;
		float normSq = ss / n + eps;
		float scale = (float) (1.0 / Math.sqrt(normSq)); // 1 / rms

		// sum_i(gradOut_i * w_i * x_i)
		float dot = 0f;
		for (int i = 0; i < n; i++)
			dot += gradOut[i] * w[i] * x[i];

		float s3OverN = (scale * scale * scale) / n;
		float[] gradX = new float[n];
		for (int i = 0; i < n; i++)
			gradX[i] = w[i] * scale * gradOut[i] - x[i] * s3OverN * dot;
		return gradX;
	}

	/**
	 * Inverse RoPE rotation: R(-angle) applied in-place to gradients.
	 * Mathematically: R^T is the rotation by -angle, same formula as forward but
	 * with negated sin.
	 */
	static void ropeBackward(float[] g, int pos, int nHeads, int headDim, float ropeTheta) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) Math.cos(angle);
				float sinA = (float) Math.sin(angle);
				float g0 = g[base + 2 * i];
				float g1 = g[base + 2 * i + 1];
				// Inverse rotation R^T: [[cos, sin], [-sin, cos]]
				g[base + 2 * i] = g0 * cosA + g1 * sinA;
				g[base + 2 * i + 1] = -g0 * sinA + g1 * cosA;
			}
		}
	}

	/**
	 * Transpose matrix–vector multiply: y[cols] = A^T × v. Dequantises one row at a
	 * time to avoid materialising the full float matrix.
	 */
	static float[] transposedMatVec(GgufReader.QuantizedTensor A, float[] v, int rows, int cols) {
		return switch (A.type()) {
		case 0 -> transposedF32(A.data(), v, rows, cols);
		case 8 -> transposedQ8_0(A.data(), v, rows, cols);
		case 12 -> transposedQ4K(A.data(), v, rows, cols);
		case 13 -> transposedQ5K(A.data(), v, rows, cols);
		case 14 -> transposedQ6K(A.data(), v, rows, cols);
		default -> transposedFallback(A, v, rows, cols);
		};
	}

	/** F32 transpose matVec — parallel scatter-reduce over row chunks. */
	private static float[] transposedF32(byte[] raw, float[] v, int rows, int cols) {
		int nT = Math.min(rows, Runtime.getRuntime().availableProcessors());
		int rpt = (rows + nT - 1) / nT;
		float[][] locals = new float[nT][cols];

		IntStream.range(0, nT).parallel().forEach(tid -> {
			int rStart = tid * rpt, rEnd = Math.min(rStart + rpt, rows);
			float[] loc = locals[tid];
			for (int r = rStart; r < rEnd; r++) {
				float vr = v[r];
				if (vr == 0f)
					continue;
				int rowOff = r * cols * 4;
				for (int c = 0; c < cols; c++) {
					int off = rowOff + c * 4;
					int bits = (raw[off] & 0xFF) | ((raw[off + 1] & 0xFF) << 8) | ((raw[off + 2] & 0xFF) << 16)
							| ((raw[off + 3] & 0xFF) << 24);
					loc[c] += Float.intBitsToFloat(bits) * vr;
				}
			}
		});
		return scatterReduce(locals, cols);
	}

	/** Q8_0 transpose matVec — parallel scatter-reduce over row chunks. */
	private static float[] transposedQ8_0(byte[] raw, float[] v, int rows, int cols) {
		final int BS = 32, BB = 34;
		final int bpr = cols / BS, bpRow = bpr * BB;
		int nT = Math.min(rows, Runtime.getRuntime().availableProcessors());
		int rpt = (rows + nT - 1) / nT;
		float[][] locals = new float[nT][cols];

		IntStream.range(0, nT).parallel().forEach(tid -> {
			int rStart = tid * rpt, rEnd = Math.min(rStart + rpt, rows);
			float[] loc = locals[tid];
			for (int r = rStart; r < rEnd; r++) {
				float vr = v[r];
				if (vr == 0f)
					continue;
				int rowOff = r * bpRow;
				for (int b = 0; b < bpr; b++) {
					int bo = rowOff + b * BB;
					float sc = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
					int cBase = b * BS;
					for (int i = 0; i < BS; i++)
						loc[cBase + i] += sc * raw[bo + 2 + i] * vr;
				}
			}
		});
		return scatterReduce(locals, cols);
	}

	/** Q4_K transpose matVec — parallel scatter-reduce over row chunks. */
	private static float[] transposedQ4K(byte[] raw, float[] v, int rows, int cols) {
		final int BLOCK_SIZE = 256, BLOCK_BYTES = 144;
		final int bpr = cols / BLOCK_SIZE, bpRow = bpr * BLOCK_BYTES;
		int nT = Math.min(rows, Runtime.getRuntime().availableProcessors());
		int rpt = (rows + nT - 1) / nT;
		float[][] locals = new float[nT][cols];

		IntStream.range(0, nT).parallel().forEach(tid -> {
			int rStart = tid * rpt, rEnd = Math.min(rStart + rpt, rows);
			float[] loc = locals[tid];
			for (int r = rStart; r < rEnd; r++) {
				float vr = v[r];
				if (vr == 0f)
					continue;
				int rowOff = r * bpRow;
				for (int b = 0; b < bpr; b++) {
					int bo = rowOff + b * BLOCK_BYTES;
					int scBase = bo + 4;
					int qsBase = bo + 16;
					float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
					float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
					int cBase = b * BLOCK_SIZE;
					int qi = 0;
					for (int g = 0; g < BLOCK_SIZE; g += 64) {
						int s0 = g / 32, s1 = s0 + 1;
						float sc0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s0);
						float mn0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s0);
						float sc1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s1);
						float mn1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s1);
						for (int i = 0; i < 32; i++) {
							loc[cBase + g + i] += (sc0 * (raw[qsBase + qi + i] & 0x0F) - mn0) * vr;
							loc[cBase + g + 32 + i] += (sc1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - mn1) * vr;
						}
						qi += 32;
					}
				}
			}
		});
		return scatterReduce(locals, cols);
	}

	/**
	 * Q5_K transpose matVec — mirrors {@code matVecQ5Kraw} in column direction.
	 * Block layout: d:f16, dmin:f16, sc[12], qh[32], qs[128] = 176 bytes per 256
	 * elements.
	 */
	private static float[] transposedQ5K(byte[] raw, float[] v, int rows, int cols) {
		final int BLOCK_SIZE = 256, BLOCK_BYTES = 176;
		final int bpr = cols / BLOCK_SIZE, bpRow = bpr * BLOCK_BYTES;
		int nT = Math.min(rows, Runtime.getRuntime().availableProcessors());
		int rpt = (rows + nT - 1) / nT;
		float[][] locals = new float[nT][cols];

		IntStream.range(0, nT).parallel().forEach(tid -> {
			int rStart = tid * rpt, rEnd = Math.min(rStart + rpt, rows);
			float[] loc = locals[tid];
			for (int r = rStart; r < rEnd; r++) {
				float vr = v[r];
				if (vr == 0f)
					continue;
				int rowOff = r * bpRow;
				for (int b = 0; b < bpr; b++) {
					int bo = rowOff + b * BLOCK_BYTES;
					int scBase = bo + 4;
					int qhBase = bo + 16;
					int qsBase = bo + 48;
					float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo));
					float dmin = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 2));
					int cBase = b * BLOCK_SIZE;
					int qi = 0;
					for (int g = 0; g < 4; g++) {
						int s0 = g * 2, s1 = s0 + 1;
						int hiBit0 = g * 2, hiBit1 = g * 2 + 1;
						float sc0 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s0);
						float mn0 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s0);
						float sc1 = d * LlamaTransformerHandler.q4kScaleRaw(raw, scBase, s1);
						float mn1 = dmin * LlamaTransformerHandler.q4kMinRaw(raw, scBase, s1);
						for (int l = 0; l < 32; l++) {
							int lo0 = raw[qsBase + qi + l] & 0x0F;
							int hi0 = (raw[qhBase + l] >>> hiBit0) & 1;
							loc[cBase + g * 64 + l] += (sc0 * (lo0 | (hi0 << 4)) - mn0) * vr;
							int lo1 = (raw[qsBase + qi + l] >>> 4) & 0x0F;
							int hi1 = (raw[qhBase + l] >>> hiBit1) & 1;
							loc[cBase + g * 64 + 32 + l] += (sc1 * (lo1 | (hi1 << 4)) - mn1) * vr;
						}
						qi += 32;
					}
				}
			}
		});
		return scatterReduce(locals, cols);
	}

	/**
	 * Q6_K transpose matVec — mirrors {@code matVecQ6Kraw} in column direction.
	 * Block layout: ql[128], qh[64], sc[16], d:f16 = 210 bytes per 256 elements.
	 * Signed 6-bit values in [-32,31], scaled by d * sc[].
	 */
	private static float[] transposedQ6K(byte[] raw, float[] v, int rows, int cols) {
		final int BLOCK_SIZE = 256, BLOCK_BYTES = 210;
		final int bpr = cols / BLOCK_SIZE, bpRow = bpr * BLOCK_BYTES;
		int nT = Math.min(rows, Runtime.getRuntime().availableProcessors());
		int rpt = (rows + nT - 1) / nT;
		float[][] locals = new float[nT][cols];

		IntStream.range(0, nT).parallel().forEach(tid -> {
			int rStart = tid * rpt, rEnd = Math.min(rStart + rpt, rows);
			float[] loc = locals[tid];
			for (int r = rStart; r < rEnd; r++) {
				float vr = v[r];
				if (vr == 0f)
					continue;
				int rowOff = r * bpRow;
				for (int b = 0; b < bpr; b++) {
					int bo = rowOff + b * BLOCK_BYTES;
					// layout: ql[128] at bo, qh[64] at bo+128, sc[16] at bo+192, d:f16 at bo+208
					float d = GgufReader.f16ToF32(LlamaTransformerHandler.readLE16(raw, bo + 208));
					int cBase = b * BLOCK_SIZE;
					for (int half = 0; half < 2; half++) {
						int qlOff = bo + half * 64;
						int qhOff = bo + 128 + half * 32;
						int scOff = bo + 192 + half * 8;
						int cOff = cBase + half * 128;
						for (int l = 0; l < 32; l++) {
							int is = l / 16;
							int qlL = raw[qlOff + l] & 0xFF;
							int qlL2 = raw[qlOff + l + 32] & 0xFF;
							int qhL = raw[qhOff + l] & 0xFF;
							int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
							int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
							int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
							int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
							float d1 = d * raw[scOff + is];
							float d2 = d * raw[scOff + is + 2];
							float d3 = d * raw[scOff + is + 4];
							float d4 = d * raw[scOff + is + 6];
							loc[cOff + l] += d1 * q1 * vr;
							loc[cOff + l + 32] += d2 * q2 * vr;
							loc[cOff + l + 64] += d3 * q3 * vr;
							loc[cOff + l + 96] += d4 * q4 * vr;
						}
					}
				}
			}
		});
		return scatterReduce(locals, cols);
	}

	/**
	 * Reduce thread-local accumulator arrays into a single result. locals[0] is
	 * reused as the output to avoid one extra allocation.
	 */
	private static float[] scatterReduce(float[][] locals, int cols) {
		float[] y = locals[0];
		for (int t = 1; t < locals.length; t++) {
			float[] loc = locals[t];
			for (int c = 0; c < cols; c++)
				y[c] += loc[c];
		}
		return y;
	}

	/**
	 * Fallback for quantisation types not yet covered by a dedicated case (e.g. F16
	 * type=1, BF16 type=30, Q4_0 type=2).
	 *
	 * <p>
	 * Parallelises over output columns: each thread computes one column of A (via a
	 * standard-basis forward matVec), then dots it with v. Correct for all
	 * quantisation types because it reuses {@link LlamaTransformerHandler#matVec}.
	 * Slower than the dedicated cases because each of the {@code cols} matVec calls
	 * processes all {@code rows} elements.
	 *
	 * <p>
	 * A WARNING is logged so you know to add a dedicated {@code transposedXxx}
	 * implementation for that type.
	 */
	private static float[] transposedFallback(GgufReader.QuantizedTensor A, float[] v, int rows, int cols) {
		java.util.logging.Logger.getLogger(LoraTrainableHandler.class.getName())
				.warning("transposedFallback: no dedicated transpose for GGML type=" + A.type() + " (" + rows + "x"
						+ cols + "). Training will be slow — add a transposedTypeXxx case.");
		float[] y = new float[cols];
		IntStream.range(0, cols).parallel().forEach(c -> {
			float[] ec = new float[cols];
			ec[c] = 1f;
			float[] column = LlamaTransformerHandler.matVec(A, ec, rows, cols);
			float acc = 0f;
			for (int r = 0; r < rows; r++)
				acc += column[r] * v[r];
			y[c] = acc;
		});
		return y;
	}

	/** Elementwise add, returns new array. */
	private static float[] add(float[] a, float[] b) {
		float[] out = new float[a.length];
		for (int i = 0; i < a.length; i++)
			out[i] = a[i] + b[i];
		return out;
	}

	/** Elementwise accumulate: dst[i] += src[i], in-place. */
	private static void addInPlace(float[] dst, float[] src) {
		for (int i = 0; i < dst.length; i++)
			dst[i] += src[i];
	}

	// ── KV cache growth ───────────────────────────────────────────────────────

	private static void ensureKvCapacity(float[][] cache, int pos, int kvDim) {
		int required = (pos + 1) * kvDim;
		for (int li = 0; li < cache.length; li++) {
			if (cache[li].length < required) {
				int newLen = cache[li].length;
				while (newLen < required)
					newLen = Math.min(newLen * 2, MAX_SEQ_LEN * kvDim);
				cache[li] = Arrays.copyOf(cache[li], newLen);
			}
		}
	}
}