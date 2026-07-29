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
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.logging.Logger;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraTrainContext;
import cab.ml.juno.lora.QaLoraAdapter;

/**
 * Qwen3 dense LoRA training and playback handler.
 *
 * <p>Mirrors {@link LoraTrainableHandler} but supports Qwen3-specific layout
 * details: {@code qDim} may exceed {@code hiddenDim}, per-head Q/K RMS norms
 * applied before RoPE, and either standard adjacent-pair RoPE or YaRN scaling
 * ({@link Qwen3Rope}). MoE and grouped-expert routing are out of scope; use
 * {@link LoraTrainingHandlerFactory} to reject the {@code qwen3moe} arch.
 */
public final class Qwen3LoraTrainableHandler implements LoraTrainingHandler {

	private static final Logger log = Logger.getLogger(Qwen3LoraTrainableHandler.class.getName());

	private record LayerState(
			float[] xIn,             // [H]
			float[] xNorm1,          // [H]
			float[] qPreNorm,        // pre-norm Q (matVec + LoRA) [qDim]
			float[] kPreNorm,        // pre-norm K [kvDim]
			float[] qPostNorm,       // after per-head rmsNorm [qDim]
			float[] kPostNorm,       // after per-head rmsNorm [kvDim]
			float[] qPostRope,       // after RoPE [qDim]
			float[][] attnW,         // [numHeads][seqLen]
			float[] attnOut,         // [qDim]
			float[] xRes2,           // xIn + attnProj [H]
			float[] xNorm2,          // [H]
			float[] gate,            // [I]
			float[] up,              // [I]
			float[] hiddenAct        // silu(gate)*up [I]
	) {
	}

	// ── Frozen weights ────────────────────────────────────────────────────────

	private final Qwen3Config cfg;
	private final int startLayer, endLayer;
	private final boolean hasEmbeddings, hasOutputProj;

	private final float[] tokenEmbd;
	private final float[] outputNorm;
	private final float[] outputProj;

	private final float[][] attnNorm; // [L][H]
	private final float[][] qNorm;    // [L][headDim]
	private final float[][] kNorm;    // [L][headDim]
	private final float[][] ffnNorm;  // [L][H]

	private final GgufReader.QuantizedTensor[] attnQ;   // [qDim, H]
	private final GgufReader.QuantizedTensor[] attnK;   // [kvDim, H]
	private final GgufReader.QuantizedTensor[] attnV;   // [kvDim, H]
	private final GgufReader.QuantizedTensor[] wo;      // [H, qDim]
	private final GgufReader.QuantizedTensor[] ffnGate; // [I, H]
	private final GgufReader.QuantizedTensor[] ffnUp;   // [I, H]
	private final GgufReader.QuantizedTensor[] wDown;   // [H, I]

	private final LoraAdapterSet loraAdapters;
	private LoraTrainContext trainCtx = LoraTrainContext.disabled();
	private int trainTokenPos;

	private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
	private static final int MAX_SEQ_LEN = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64;

	// ── Factory ───────────────────────────────────────────────────────────────

	public static Qwen3LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		return load(modelPath, context, adapters, CpuMatVec.INSTANCE);
	}

	/**
	 * Load a Qwen3 shard with LoRA adapters. The {@code backend} argument is
	 * accepted for signature symmetry with {@link LoraTrainingHandlerFactory};
	 * this Tier 6 implementation always runs CPU quantised matmul.
	 */
	public static Qwen3LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		log.info("Loading Qwen3 LoRA handler: layers " + context.startLayer() + "-" + context.endLayer()
				+ "  adapters=" + adapters.size() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			Qwen3Config cfg = Qwen3Config.from(r);
			LoraModelLayout layout = LoraModelLayout.qwen3(cfg);
			LoraInitializer.validate(adapters, layout);
			return new Qwen3LoraTrainableHandler(r, cfg, context, adapters);
		}
	}

	private Qwen3LoraTrainableHandler(GgufReader r, Qwen3Config cfg, ShardContext ctx, LoraAdapterSet adapters)
			throws IOException {
		this.cfg = cfg;
		this.loraAdapters = adapters;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L = endLayer - startLayer;
		int headDim = cfg.headDim();

		this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

		attnNorm = new float[L][];
		qNorm = new float[L][];
		kNorm = new float[L][];
		ffnNorm = new float[L][];
		attnQ = new GgufReader.QuantizedTensor[L];
		attnK = new GgufReader.QuantizedTensor[L];
		attnV = new GgufReader.QuantizedTensor[L];
		wo = new GgufReader.QuantizedTensor[L];
		ffnGate = new GgufReader.QuantizedTensor[L];
		ffnUp = new GgufReader.QuantizedTensor[L];
		wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			qNorm[li] = r.tensor("blk." + i + ".attn_q_norm.weight");
			kNorm[li] = r.tensor("blk." + i + ".attn_k_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");
			if (qNorm[li].length != headDim || kNorm[li].length != headDim)
				throw new IOException(
						"Layer " + i + ": qNorm/kNorm length must equal headDim=" + headDim);

			attnQ[li] = r.tensorRaw("blk." + i + ".attn_q.weight");
			attnK[li] = r.tensorRaw("blk." + i + ".attn_k.weight");
			attnV[li] = r.tensorRaw("blk." + i + ".attn_v.weight");
			wo[li] = r.tensorRaw("blk." + i + ".attn_output.weight");
			ffnGate[li] = r.tensorRaw("blk." + i + ".ffn_gate.weight");
			ffnUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");
		}
	}

	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight"))
			return r.tensor("output.weight");
		log.info("output.weight absent — using tied embeddings");
		return r.tensor("token_embd.weight");
	}

	// ── ForwardPassHandler (inference) ────────────────────────────────────────

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long t0 = System.nanoTime();
		ForwardPassEvent evt = new ForwardPassEvent();
		evt.begin();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		ForwardResult result;
		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			result = ForwardResult.logits(request.requestId(), logits, System.nanoTime() - t0);
		} else {
			result = ForwardResult.activations(request.requestId(), x, System.nanoTime() - t0);
		}

		evt.handlerType = "qwen3-lora";
		evt.requestId = request.requestId();
		evt.startPosition = request.startPosition();
		evt.layerCount = endLayer - startLayer;
		evt.hasOutputProjection = hasOutputProj;
		evt.commit();
		return result;
	}

	@Override
	public Optional<float[]> lastRmsHiddenForEmbedding(ForwardRequest request, ShardContext context) {
		if (!hasOutputProj)
			return Optional.empty();
		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());
		return Optional.of(LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps()));
	}

	@Override
	public boolean isReady() {
		return true;
	}

	@Override
	public void releaseGpuResources() {
		// CPU-only Tier 6 implementation.
	}

	private float[] getInitialActivation(ForwardRequest req) {
		if (hasEmbeddings) {
			int[] ids = req.tokenIds();
			int actualVocab = tokenEmbd.length / cfg.hiddenDim();
			int id = Math.max(0, Math.min(ids[ids.length - 1], actualVocab - 1));
			float[] x = new float[cfg.hiddenDim()];
			System.arraycopy(tokenEmbd, id * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
			return x;
		}
		return req.activations().clone();
	}

	private float[] runLayers(float[] x, String requestId, int pos) {
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();
		kvCacheK.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		kvCacheV.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		float[][] kC = kvCacheK.get(requestId);
		float[][] vC = kvCacheV.get(requestId);
		ensureKvCapacity(kC, pos, kvDim);
		ensureKvCapacity(vC, pos, kvDim);
		for (int li = 0; li < L; li++)
			x = inferenceLayer(x, li, pos, kC[li], vC[li]);
		return x;
	}

	private float[] inferenceLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[] q = LlamaTransformerHandler.matVec(attnQ[li], xNorm1, qDim, H);
		float[] k = LlamaTransformerHandler.matVec(attnK[li], xNorm1, kvDim, H);
		float[] v = LlamaTransformerHandler.matVec(attnV[li], xNorm1, kvDim, H);

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		Qwen3TransformerHandler.rmsNormPerHead(q, qNorm[li], cfg.numHeads(), cfg.headDim(), cfg.rmsNormEps());
		Qwen3TransformerHandler.rmsNormPerHead(k, kNorm[li], cfg.numKvHeads(), cfg.headDim(), cfg.rmsNormEps());

		Qwen3Rope.apply(q, pos, cfg.numHeads(), cfg.headDim(), cfg.rope());
		Qwen3Rope.apply(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.rope());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut = Qwen3TransformerHandler.gqa(cfg, q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, H, qDim);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	private float[] outputProjection(float[] x) {
		float[] xn = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xn, actualVocab, cfg.hiddenDim());
	}

	private float[] ffn(float[] xNorm2, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = LlamaTransformerHandler.matVec(ffnGate[li], xNorm2, I, H);
		float[] up = LlamaTransformerHandler.matVec(ffnUp[li], xNorm2, I, H);
		applyLoraInPlace(gate, li, "wgate", xNorm2);
		applyLoraInPlace(up, li, "wup", xNorm2);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		float[] down = LlamaTransformerHandler.matVec(wDown[li], hidden, H, I);
		applyLoraInPlace(down, li, "wdown", hidden);
		return down;
	}

	// ── Training step ─────────────────────────────────────────────────────────

	@Override
	public LoraGradientResult computeGradients(int[] tokens) {
		return computeGradients(tokens, null, LoraTrainContext.disabled());
	}

	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask) {
		return computeGradients(tokens, lossMask, LoraTrainContext.disabled());
	}

	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask, LoraTrainContext ctx) {
		if (tokens.length < 2)
			throw new IllegalArgumentException("tokens.length must be >= 2");
		int T = tokens.length - 1;
		if (lossMask != null && lossMask.length != T)
			throw new IllegalArgumentException("lossMask.length must equal tokens.length - 1");

		trainCtx = ctx != null ? ctx : LoraTrainContext.disabled();
		try {
			int L = endLayer - startLayer;
			int H = cfg.hiddenDim();
			int kvDim = cfg.kvDim();

			LayerState[][] allStates = new LayerState[T][L];
			float[][] allXFinal = new float[T][];
			float[][] allXNormFinal = new float[T][];
			float[][] allProbs = new float[T][];

			float[][] kCache = new float[L][T * kvDim];
			float[][] vCache = new float[L][T * kvDim];

			long t0 = System.currentTimeMillis();
			for (int pos = 0; pos < T; pos++) {
				trainTokenPos = pos;
				float[] x = embedding(tokens[pos]);
				for (int li = 0; li < L; li++) {
					allStates[pos][li] = forwardLayerStore(x, li, pos, kCache[li], vCache[li]);
					x = computeLayerOutput(allStates[pos][li], li);
				}
				if (hasOutputProj) {
					allXFinal[pos] = x.clone();
					allXNormFinal[pos] = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
					int actualVocab = outputProj.length / H;
					float[] logits = LlamaTransformerHandler.matVec(outputProj, allXNormFinal[pos], actualVocab, H);
					allProbs[pos] = softmaxCopy(logits);
				}
			}
			long forwardMs = System.currentTimeMillis() - t0;

			float lossSum = 0f;
			int predictionCount = 0;
			for (int pos = 0; pos < T; pos++) {
				if (lossMask != null && !lossMask[pos])
					continue;
				int target = tokens[pos + 1];
				if (allProbs[pos] != null)
					lossSum -= (float) Math.log(Math.max(allProbs[pos][target], 1e-9f));
				predictionCount++;
			}

			long t1 = System.currentTimeMillis();
			int actualVocab = hasOutputProj ? outputProj.length / H : 0;
			for (int pos = 0; pos < T; pos++) {
				if (lossMask != null && !lossMask[pos])
					continue;
				trainTokenPos = pos;
				int target = tokens[pos + 1];

				float[] gradX;
				if (hasOutputProj) {
					float[] gradLogits = allProbs[pos].clone();
					gradLogits[target] -= 1.0f;
					float[] gradXNormFinal = LoraTrainingMath.transposedMatVec(
							new GgufReader.QuantizedTensor("output", 0, actualVocab * H, floatBytes(outputProj)),
							gradLogits, actualVocab, H);
					gradX = LoraTrainingMath.rmsNormBackward(allXFinal[pos], outputNorm, gradXNormFinal,
							cfg.rmsNormEps());
				} else {
					gradX = new float[H];
				}

				for (int li = L - 1; li >= 0; li--) {
					gradX = backwardLayer(gradX, li, pos, allStates[pos][li], kCache[li], vCache[li], pos + 1);
				}
			}
			long backwardMs = System.currentTimeMillis() - t1;

			return new LoraGradientResult(lossSum, predictionCount, forwardMs, backwardMs);
		} finally {
			trainCtx = LoraTrainContext.disabled();
		}
	}

	@Override
	public LoraGradientResult evaluateLoss(int[] tokens) {
		return evaluateLoss(tokens, null);
	}

	@Override
	public LoraGradientResult evaluateLoss(int[] tokens, boolean[] lossMask) {
		if (tokens.length < 2)
			throw new IllegalArgumentException("tokens.length must be >= 2");
		int T = tokens.length - 1;
		if (lossMask != null && lossMask.length != T)
			throw new IllegalArgumentException("lossMask.length must equal tokens.length - 1");

		trainCtx = LoraTrainContext.disabled();
		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		float[][] kCache = new float[L][T * kvDim];
		float[][] vCache = new float[L][T * kvDim];

		long t0 = System.currentTimeMillis();
		float lossSum = 0f;
		int predictionCount = 0;
		for (int pos = 0; pos < T; pos++) {
			float[] x = embedding(tokens[pos]);
			for (int li = 0; li < L; li++)
				x = inferenceLayer(x, li, pos, kCache[li], vCache[li]);
			if (lossMask != null && !lossMask[pos])
				continue;
			if (hasOutputProj) {
				float[] xn = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
				int actualVocab = outputProj.length / H;
				float[] logits = LlamaTransformerHandler.matVec(outputProj, xn, actualVocab, H);
				float[] probs = softmaxCopy(logits);
				lossSum -= (float) Math.log(Math.max(probs[tokens[pos + 1]], 1e-9f));
			}
			predictionCount++;
		}
		long forwardMs = System.currentTimeMillis() - t0;
		return new LoraGradientResult(lossSum, predictionCount, forwardMs, 0L);
	}

	@Override
	public LoraAdapterSet adapters() {
		return loraAdapters;
	}

	@Override
	public String architecture() {
		return cfg.architecture();
	}

	@Override
	public LoraModelLayout layout() {
		return LoraModelLayout.qwen3(cfg);
	}

	public Qwen3Config config() {
		return cfg;
	}

	// ── Training forward with capture ─────────────────────────────────────────

	private float[] embedding(int tokenId) {
		if (!hasEmbeddings)
			throw new IllegalStateException("shard does not own embeddings");
		int actualVocab = tokenEmbd.length / cfg.hiddenDim();
		tokenId = Math.max(0, Math.min(tokenId, actualVocab - 1));
		float[] x = new float[cfg.hiddenDim()];
		System.arraycopy(tokenEmbd, tokenId * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
		return x;
	}

	private LayerState forwardLayerStore(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();
		int NH = cfg.numHeads();
		int NKV = cfg.numKvHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[] q = LlamaTransformerHandler.matVec(attnQ[li], xNorm1, qDim, H);
		float[] k = LlamaTransformerHandler.matVec(attnK[li], xNorm1, kvDim, H);
		float[] v = LlamaTransformerHandler.matVec(attnV[li], xNorm1, kvDim, H);

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		float[] qPreNorm = q.clone();
		float[] kPreNorm = k.clone();

		Qwen3TransformerHandler.rmsNormPerHead(q, qNorm[li], NH, Hd, cfg.rmsNormEps());
		Qwen3TransformerHandler.rmsNormPerHead(k, kNorm[li], NKV, Hd, cfg.rmsNormEps());
		float[] qPostNorm = q.clone();
		float[] kPostNorm = k.clone();

		Qwen3Rope.apply(q, pos, NH, Hd, cfg.rope());
		Qwen3Rope.apply(k, pos, NKV, Hd, cfg.rope());
		float[] qPostRope = q.clone();

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		int seqLen = pos + 1;
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] attnOut = new float[qDim];
		float[][] attnW = new float[NH][seqLen];
		float[] scores = new float[seqLen];
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
			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOff = t * kvDim + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++)
					attnOut[outBase + d] += w * vCacheLayer[vOff + d];
			}
		}

		float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, H, qDim);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] xRes2 = LlamaTransformerHandler.add(x, attnProj);
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(xRes2, ffnNorm[li], cfg.rmsNormEps());

		int I = cfg.intermediateSize();
		float[] gate = LlamaTransformerHandler.matVec(ffnGate[li], xNorm2, I, H);
		float[] up = LlamaTransformerHandler.matVec(ffnUp[li], xNorm2, I, H);
		applyLoraInPlace(gate, li, "wgate", xNorm2);
		applyLoraInPlace(up, li, "wup", xNorm2);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];

		return new LayerState(x.clone(), xNorm1, qPreNorm, kPreNorm, qPostNorm, kPostNorm, qPostRope, attnW, attnOut,
				xRes2, xNorm2, gate, up, hidden);
	}

	private float[] computeLayerOutput(LayerState st, int li) {
		int H = cfg.hiddenDim();
		float[] ffnOut = LlamaTransformerHandler.matVec(wDown[li], st.hiddenAct(), H, cfg.intermediateSize());
		applyLoraInPlace(ffnOut, li, "wdown", st.hiddenAct());
		return LlamaTransformerHandler.add(st.xRes2(), ffnOut);
	}

	// ── Backward ──────────────────────────────────────────────────────────────

	private float[] backwardLayer(float[] gradOut, int li, int pos, LayerState st, float[] kCacheLayer,
			float[] vCacheLayer, int seqLen) {
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int I = cfg.intermediateSize();
		int kvDim = cfg.kvDim();
		int NH = cfg.numHeads();
		int NKV = cfg.numKvHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();
		int absLayer = li + startLayer;

		// ── FFN residual ─────────────────────────────────────────────────────
		float[] gradXRes2 = gradOut.clone();
		float[] gradFfnOut = gradOut;

		float[] gradHidden = LoraTrainingMath.transposedMatVec(wDown[li], gradFfnOut, H, I);
		addLoraBackward(gradHidden, absLayer, "wdown", gradFfnOut, st.hiddenAct());

		float[] gradGate = new float[I];
		float[] gradUp = new float[I];
		for (int i = 0; i < I; i++) {
			float g = st.gate()[i];
			float sig = 1f / (1f + (float) Math.exp(-g));
			gradGate[i] = gradHidden[i] * st.up()[i] * sig * (1f + g * (1f - sig));
			gradUp[i] = gradHidden[i] * LlamaTransformerHandler.silu(g);
		}

		float[] gradXNorm2 = add(
				LoraTrainingMath.transposedMatVec(ffnGate[li], gradGate, I, H),
				LoraTrainingMath.transposedMatVec(ffnUp[li], gradUp, I, H));
		addLoraBackward(gradXNorm2, absLayer, "wgate", gradGate, st.xNorm2());
		addLoraBackward(gradXNorm2, absLayer, "wup", gradUp, st.xNorm2());

		addInPlace(gradXRes2, LoraTrainingMath.rmsNormBackward(st.xRes2(), ffnNorm[li], gradXNorm2, cfg.rmsNormEps()));

		// ── Attention residual ───────────────────────────────────────────────
		float[] gradXIn = gradXRes2.clone();
		float[] gradAttnProj = gradXRes2;

		float[] gradAttnOut = LoraTrainingMath.transposedMatVec(wo[li], gradAttnProj, H, qDim);
		addLoraBackward(gradAttnOut, absLayer, "wo", gradAttnProj, st.attnOut());

		// ── Attention backward ────────────────────────────────────────────────
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] gradQPostRope = new float[qDim];
		float[] gradK = new float[kvDim]; // gradient at current position only (truncated BPTT)
		float[] gradV = new float[kvDim];

		for (int h = 0; h < NH; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;
			float[] aw = st.attnW()[h];

			float[] gradAttnOut_h = Arrays.copyOfRange(gradAttnOut, qBase, qBase + Hd);

			for (int d = 0; d < Hd; d++)
				gradV[kBase + d] += aw[pos] * gradAttnOut_h[d];

			float[] dotWithV = new float[seqLen];
			for (int t = 0; t < seqLen; t++) {
				int vOff = t * kvDim + kBase;
				float d2 = 0f;
				for (int d = 0; d < Hd; d++)
					d2 += gradAttnOut_h[d] * vCacheLayer[vOff + d];
				dotWithV[t] = d2;
			}
			float sumDot = 0f;
			for (int t = 0; t < seqLen; t++)
				sumDot += aw[t] * dotWithV[t];
			float[] gradScores = new float[seqLen];
			for (int t = 0; t < seqLen; t++)
				gradScores[t] = aw[t] * (dotWithV[t] - sumDot);

			for (int t = 0; t < seqLen; t++) {
				if (gradScores[t] == 0f)
					continue;
				int kOff = t * kvDim + kBase;
				float gs = gradScores[t] * scale;
				for (int d = 0; d < Hd; d++)
					gradQPostRope[qBase + d] += gs * kCacheLayer[kOff + d];
			}
			float gsPos = gradScores[pos] * scale;
			if (gsPos != 0f) {
				for (int d = 0; d < Hd; d++)
					gradK[kBase + d] += gsPos * st.qPostRope()[qBase + d];
			}
		}

		Qwen3Rope.applyBackward(gradQPostRope, pos, NH, Hd, cfg.rope());
		Qwen3Rope.applyBackward(gradK, pos, NKV, Hd, cfg.rope());

		// Per-head RMS norm adjoint for Q and K.
		float[] gradQ = LoraTrainingMath.perHeadRmsNormBackward(st.qPreNorm(), qNorm[li], gradQPostRope, NH, Hd,
				cfg.rmsNormEps());
		float[] gradKPre = LoraTrainingMath.perHeadRmsNormBackward(st.kPreNorm(), kNorm[li], gradK, NKV, Hd,
				cfg.rmsNormEps());

		float[] gradXNorm1 = LoraTrainingMath.transposedMatVec(attnQ[li], gradQ, qDim, H);
		addLoraBackward(gradXNorm1, absLayer, "wq", gradQ, st.xNorm1());

		float[] gradXNorm1_k = LoraTrainingMath.transposedMatVec(attnK[li], gradKPre, kvDim, H);
		addLoraBackward(gradXNorm1_k, absLayer, "wk", gradKPre, st.xNorm1());
		addInPlace(gradXNorm1, gradXNorm1_k);

		float[] gradXNorm1_v = LoraTrainingMath.transposedMatVec(attnV[li], gradV, kvDim, H);
		addLoraBackward(gradXNorm1_v, absLayer, "wv", gradV, st.xNorm1());
		addInPlace(gradXNorm1, gradXNorm1_v);

		addInPlace(gradXIn, LoraTrainingMath.rmsNormBackward(st.xIn(), attnNorm[li], gradXNorm1, cfg.rmsNormEps()));
		return gradXIn;
	}

	// ── LoRA helpers ──────────────────────────────────────────────────────────

	private void applyLoraInPlace(float[] out, int li, String proj, float[] input) {
		int absLayer = li + startLayer;
		QaLoraAdapter qa = loraAdapters.getQa(absLayer, proj);
		if (qa != null) {
			float[] delta;
			if (trainCtx.dropoutEnabled()) {
				int projOrd = LoraProjection.fromKey(proj).ordinal();
				delta = qa.forwardTrain(input, trainCtx.dropoutRate(), trainCtx.rootSeed(), trainCtx.optimizerUpdate(),
						trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
			} else {
				delta = qa.forward(input);
			}
			for (int i = 0; i < out.length; i++)
				out[i] += delta[i];
			return;
		}
		LoraAdapter lora = loraAdapters.get(absLayer, proj);
		if (lora == null)
			return;
		float[] delta;
		if (trainCtx.dropoutEnabled()) {
			int projOrd = LoraProjection.fromKey(proj).ordinal();
			delta = lora.forwardTrain(input, trainCtx.dropoutRate(), trainCtx.rootSeed(), trainCtx.optimizerUpdate(),
					trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
		} else {
			delta = lora.forward(input);
		}
		for (int i = 0; i < out.length; i++)
			out[i] += delta[i];
	}

	private void addLoraBackward(float[] dest, int absLayer, String proj, float[] gradDelta, float[] input) {
		QaLoraAdapter qa = loraAdapters.getQa(absLayer, proj);
		if (qa != null) {
			float[] g;
			if (trainCtx.dropoutEnabled()) {
				int projOrd = LoraProjection.fromKey(proj).ordinal();
				g = qa.backwardTrain(gradDelta, input, trainCtx.dropoutRate(), trainCtx.rootSeed(),
						trainCtx.optimizerUpdate(), trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
			} else {
				g = qa.backward(gradDelta, input);
			}
			addInPlace(dest, g);
			return;
		}
		LoraAdapter lora = loraAdapters.get(absLayer, proj);
		if (lora == null)
			return;
		float[] g;
		if (trainCtx.dropoutEnabled()) {
			int projOrd = LoraProjection.fromKey(proj).ordinal();
			g = lora.backwardTrain(gradDelta, input, trainCtx.dropoutRate(), trainCtx.rootSeed(),
					trainCtx.optimizerUpdate(), trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
		} else {
			g = lora.backward(gradDelta, input);
		}
		addInPlace(dest, g);
	}

	// ── Math helpers ──────────────────────────────────────────────────────────

	private static float[] softmaxCopy(float[] logits) {
		float[] out = logits.clone();
		LlamaTransformerHandler.softmax(out, out.length);
		return out;
	}

	private static float[] add(float[] a, float[] b) {
		float[] out = new float[a.length];
		for (int i = 0; i < a.length; i++)
			out[i] = a[i] + b[i];
		return out;
	}

	private static void addInPlace(float[] dst, float[] src) {
		for (int i = 0; i < dst.length; i++)
			dst[i] += src[i];
	}

	private static byte[] floatBytes(float[] data) {
		java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(data.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
		for (float f : data)
			buf.putFloat(f);
		return buf.array();
	}

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
