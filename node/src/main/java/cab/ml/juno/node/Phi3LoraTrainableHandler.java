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
 * Phi-3 LoRA training and playback handler.
 *
 * <p>Mirrors {@link LoraTrainableHandler} but uses Phi-3's fused
 * {@code attn_qkv.weight} and {@code ffn_up.weight} tensors together with
 * NeoX-style RoPE and per-dimension rope factors ({@link Phi3Rope}).
 * Adapters are applied to logical slices via row-range matVec on the fused
 * quantised weights; backward concatenates logical slice gradients and
 * dispatches one adjoint matVec per physical tensor.
 *
 * <p>Truncated BPTT (no KV cross-position gradient). When the MatVec backend
 * is a {@link GpuMatVec}, physical fused projections are uploaded via
 * {@link LoraResidentWeights} for resident forward and transpose; otherwise
 * CPU quantised matmul. DoRA is not enabled in the Phi-3 variant.
 */
public final class Phi3LoraTrainableHandler implements LoraTrainingHandler {

	private static final Logger log = Logger.getLogger(Phi3LoraTrainableHandler.class.getName());

	private record LayerState(
			float[] xIn,        // residual stream before this layer [H]
			float[] xNorm1,     // after pre-attention rmsNorm [H]
			float[] qPostRope,  // Q after RoPE [numHeads*headDim = H]
			float[][] attnW,    // attention weights per head [numHeads][seqLen]
			float[] attnOut,    // attention output before wo [H]
			float[] xRes2,      // xIn + attnProj [H]
			float[] xNorm2,     // after pre-FFN rmsNorm [H]
			float[] gate,       // FFN gate output [I]
			float[] up,         // FFN up output [I]
			float[] hiddenAct   // silu(gate) * up [I]
	) {
	}

	// ── Frozen weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final Phi3RopeConfig ropeCfg;
	private final int startLayer, endLayer;
	private final boolean hasEmbeddings, hasOutputProj;

	private final float[] tokenEmbd;
	private final float[] outputNorm;
	private final float[] outputProj;

	private final float[][] attnNorm; // [L][H]
	private final float[][] ffnNorm;  // [L][H]

	// Fused QKV: [H + kvDim + kvDim, H]. wo: [H, H]. Fused gate+up: [2I, H]. wDown: [H, I].
	private final GgufReader.QuantizedTensor[] attnQkv;
	private final GgufReader.QuantizedTensor[] wo;
	private final GgufReader.QuantizedTensor[] ffnGateUp;
	private final GgufReader.QuantizedTensor[] wDown;

	private ResidentWeightMatrix[] attnQkvDev;
	private ResidentWeightMatrix[] woDev;
	private ResidentWeightMatrix[] ffnGateUpDev;
	private ResidentWeightMatrix[] wDownDev;
	private ResidentWeightMatrix outputProjDev;

	// ── LoRA adapters ─────────────────────────────────────────────────────────

	private final LoraAdapterSet loraAdapters;
	private LoraTrainContext trainCtx = LoraTrainContext.disabled();
	private int trainTokenPos;

	// ── Inference KV cache ────────────────────────────────────────────────────

	private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
	private static final int MAX_SEQ_LEN = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64;

	// ── Factory ───────────────────────────────────────────────────────────────

	public static Phi3LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		return load(modelPath, context, adapters, ForwardPassHandlerLoader.selectLoraBackend());
	}

	/**
	 * Load a Phi-3 shard with LoRA adapters. GPU backends upload physical fused
	 * projections via {@link LoraResidentWeights}; CPU keeps quantised matmul.
	 */
	public static Phi3LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		log.info("Loading Phi-3 LoRA handler: layers " + context.startLayer() + "-" + context.endLayer()
				+ "  adapters=" + adapters.size() + "  backend=" + backend.getClass().getSimpleName()
				+ "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			LoraModelLayout layout = LoraModelLayout.phi3(cfg);
			LoraInitializer.validate(adapters, layout);
			return new Phi3LoraTrainableHandler(r, cfg, context, adapters, backend);
		}
	}

	private Phi3LoraTrainableHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		this.cfg = cfg;
		this.ropeCfg = Phi3RopeConfig.from(r, cfg);
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
		attnQkv = new GgufReader.QuantizedTensor[L];
		wo = new GgufReader.QuantizedTensor[L];
		ffnGateUp = new GgufReader.QuantizedTensor[L];
		wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");
			attnQkv[li] = r.tensorRaw("blk." + i + ".attn_qkv.weight");
			wo[li] = r.tensorRaw("blk." + i + ".attn_output.weight");
			ffnGateUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");
		}

		attnQkvDev = woDev = ffnGateUpDev = wDownDev = null;
		outputProjDev = null;
		if (backend instanceof GpuMatVec gpu)
			uploadResidentWeights(gpu, L);
	}

	private void uploadResidentWeights(GpuMatVec gpu, int L) {
		boolean half = gpu.supportsHalfResident();
		log.info("Phi-3 LoRA: uploading fused projection weights to GPU ("
				+ (half ? "FP16" : "FP32") + ")…");
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();
		ResidentWeightMatrix[] qkvD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] woD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] gateUpD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] downD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] outHolder = new ResidentWeightMatrix[1];
		try {
			for (int li = 0; li < L; li++) {
				qkvD[li] = LoraResidentWeights.uploadQuant(gpu, attnQkv[li], H + 2 * kvDim, H);
				woD[li] = LoraResidentWeights.uploadQuant(gpu, wo[li], H, H);
				gateUpD[li] = LoraResidentWeights.uploadQuant(gpu, ffnGateUp[li], 2 * I, H);
				downD[li] = LoraResidentWeights.uploadQuant(gpu, wDown[li], H, I);
			}
			if (outputProj != null) {
				int V = outputProj.length / H;
				outHolder[0] = LoraResidentWeights.upload(gpu, outputProj, V, H);
			}
			this.attnQkvDev = qkvD;
			this.woDev = woD;
			this.ffnGateUpDev = gateUpD;
			this.wDownDev = downD;
			this.outputProjDev = outHolder[0];
			log.info("Phi-3 LoRA: GPU weight upload complete (" + (half ? "FP16" : "FP32") + ").");
		} catch (IllegalStateException ex) {
			LoraResidentWeights.tryRecoverFromUploadOom(ex, log, () -> {
				LoraResidentWeights.closeArray(qkvD);
				LoraResidentWeights.closeArray(woD);
				LoraResidentWeights.closeArray(gateUpD);
				LoraResidentWeights.closeArray(downD);
				LoraResidentWeights.closeQuietly(outHolder[0]);
			});
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

		evt.handlerType = "phi3-lora";
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
		LoraResidentWeights.closeArray(attnQkvDev);
		LoraResidentWeights.closeArray(woDev);
		LoraResidentWeights.closeArray(ffnGateUpDev);
		LoraResidentWeights.closeArray(wDownDev);
		LoraResidentWeights.closeQuietly(outputProjDev);
		attnQkvDev = woDev = ffnGateUpDev = wDownDev = null;
		outputProjDev = null;
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
		int kvDim = cfg.kvDim();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[][] qkv = fusedQkv(li, xNorm1);
		float[] q = qkv[0];
		float[] k = qkv[1];
		float[] v = qkv[2];

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		Phi3Rope.ropeExt(q, pos, cfg.numHeads(), cfg.headDim(), ropeCfg);
		Phi3Rope.ropeExt(k, pos, cfg.numKvHeads(), cfg.headDim(), ropeCfg);

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = LoraResidentWeights.matVec(wo[li], woDev != null ? woDev[li] : null, attnOut, H, H);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	private float[] outputProjection(float[] x) {
		float[] xn = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LoraResidentWeights.matVecDense(outputProj, outputProjDev, xn, actualVocab, cfg.hiddenDim());
	}

	private float[] ffn(float[] xNorm2, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[][] gateUp = fusedGateUp(li, xNorm2);
		float[] gate = gateUp[0];
		float[] up = gateUp[1];
		applyLoraInPlace(gate, li, "wgate", xNorm2);
		applyLoraInPlace(up, li, "wup", xNorm2);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		float[] down = LoraResidentWeights.matVec(wDown[li], wDownDev != null ? wDownDev[li] : null, hidden, H, I);
		applyLoraInPlace(down, li, "wdown", hidden);
		return down;
	}

	/** Physical fused QKV: one resident sgemv + slice, or CPU row-range matVec. */
	private float[][] fusedQkv(int li, float[] xNorm1) {
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		ResidentWeightMatrix dev = attnQkvDev != null ? attnQkvDev[li] : null;
		if (dev != null) {
			float[] qkv = dev.sgemv(xNorm1);
			return new float[][] {
					Arrays.copyOfRange(qkv, 0, H),
					Arrays.copyOfRange(qkv, H, H + kvDim),
					Arrays.copyOfRange(qkv, H + kvDim, H + 2 * kvDim)
			};
		}
		return new float[][] {
				LlamaTransformerHandler.matVec(attnQkv[li], xNorm1, 0, H, H),
				LlamaTransformerHandler.matVec(attnQkv[li], xNorm1, H, H + kvDim, H),
				LlamaTransformerHandler.matVec(attnQkv[li], xNorm1, H + kvDim, H + 2 * kvDim, H)
		};
	}

	/** Physical fused gate+up: one resident sgemv + slice, or CPU row-range matVec. */
	private float[][] fusedGateUp(int li, float[] xNorm2) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		ResidentWeightMatrix dev = ffnGateUpDev != null ? ffnGateUpDev[li] : null;
		if (dev != null) {
			float[] gateUp = dev.sgemv(xNorm2);
			return new float[][] {
					Arrays.copyOfRange(gateUp, 0, I),
					Arrays.copyOfRange(gateUp, I, 2 * I)
			};
		}
		return new float[][] {
				LlamaTransformerHandler.matVec(ffnGateUp[li], xNorm2, 0, I, H),
				LlamaTransformerHandler.matVec(ffnGateUp[li], xNorm2, I, 2 * I, H)
		};
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
					float[] logits = LoraResidentWeights.matVecDense(outputProj, outputProjDev, allXNormFinal[pos],
							actualVocab, H);
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
					float[] gradXNormFinal = LoraResidentWeights.transposedMatVecDense(outputProj, outputProjDev,
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
				float[] logits = LoraResidentWeights.matVecDense(outputProj, outputProjDev, xn, actualVocab, H);
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
		return LoraModelLayout.phi3(cfg);
	}

	public LlamaConfig config() {
		return cfg;
	}

	Phi3RopeConfig ropeConfig() {
		return ropeCfg;
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
		int kvDim = cfg.kvDim();
		int NH = cfg.numHeads();
		int Hd = cfg.headDim();

		float[] xNorm1 = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[][] qkv = fusedQkv(li, xNorm1);
		float[] q = qkv[0];
		float[] k = qkv[1];
		float[] v = qkv[2];

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);

		Phi3Rope.ropeExt(q, pos, cfg.numHeads(), Hd, ropeCfg);
		Phi3Rope.ropeExt(k, pos, cfg.numKvHeads(), Hd, ropeCfg);

		float[] qPostRope = q.clone();
		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

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

		float[] attnProj = LoraResidentWeights.matVec(wo[li], woDev != null ? woDev[li] : null, attnOut, H, H);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] xRes2 = LlamaTransformerHandler.add(x, attnProj);
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(xRes2, ffnNorm[li], cfg.rmsNormEps());

		int I = cfg.intermediateSize();
		float[][] gateUp = fusedGateUp(li, xNorm2);
		float[] gate = gateUp[0];
		float[] up = gateUp[1];
		applyLoraInPlace(gate, li, "wgate", xNorm2);
		applyLoraInPlace(up, li, "wup", xNorm2);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];

		return new LayerState(x.clone(), xNorm1, qPostRope, attnW, attnOut, xRes2, xNorm2, gate, up, hidden);
	}

	private float[] computeLayerOutput(LayerState st, int li) {
		int H = cfg.hiddenDim();
		float[] ffnOut = LoraResidentWeights.matVec(wDown[li], wDownDev != null ? wDownDev[li] : null, st.hiddenAct(),
				H, cfg.intermediateSize());
		applyLoraInPlace(ffnOut, li, "wdown", st.hiddenAct());
		return LlamaTransformerHandler.add(st.xRes2(), ffnOut);
	}

	// ── Backward ──────────────────────────────────────────────────────────────

	private float[] backwardLayer(float[] gradOut, int li, int pos, LayerState st, float[] kCacheLayer,
			float[] vCacheLayer, int seqLen) {
		int H = cfg.hiddenDim();
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

		float[] gradHidden = LoraResidentWeights.transposedMatVec(wDown[li],
				wDownDev != null ? wDownDev[li] : null, gradFfnOut, H, I);
		addLoraBackward(gradHidden, absLayer, "wdown", gradFfnOut, st.hiddenAct());

		float[] gradGate = new float[I];
		float[] gradUp = new float[I];
		for (int i = 0; i < I; i++) {
			float g = st.gate()[i];
			float sig = 1f / (1f + (float) Math.exp(-g));
			gradGate[i] = gradHidden[i] * st.up()[i] * sig * (1f + g * (1f - sig));
			gradUp[i] = gradHidden[i] * LlamaTransformerHandler.silu(g);
		}

		// One transpose matVec against the fused [2I,H] tensor.
		float[] gradGateUp = new float[2 * I];
		System.arraycopy(gradGate, 0, gradGateUp, 0, I);
		System.arraycopy(gradUp, 0, gradGateUp, I, I);
		float[] gradXNorm2 = LoraResidentWeights.transposedMatVec(ffnGateUp[li],
				ffnGateUpDev != null ? ffnGateUpDev[li] : null, gradGateUp, 2 * I, H);
		addLoraBackward(gradXNorm2, absLayer, "wgate", gradGate, st.xNorm2());
		addLoraBackward(gradXNorm2, absLayer, "wup", gradUp, st.xNorm2());

		addInPlace(gradXRes2, LoraTrainingMath.rmsNormBackward(st.xRes2(), ffnNorm[li], gradXNorm2, cfg.rmsNormEps()));

		// ── Attention residual ───────────────────────────────────────────────
		float[] gradXIn = gradXRes2.clone();
		float[] gradAttnProj = gradXRes2;

		float[] gradAttnOut = LoraResidentWeights.transposedMatVec(wo[li], woDev != null ? woDev[li] : null,
				gradAttnProj, H, H);
		addLoraBackward(gradAttnOut, absLayer, "wo", gradAttnProj, st.attnOut());

		// ── Attention backward ────────────────────────────────────────────────
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] gradQ = new float[H];
		float[] gradK = new float[kvDim];
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
					gradQ[qBase + d] += gs * kCacheLayer[kOff + d];
			}
			float gsPos = gradScores[pos] * scale;
			if (gsPos != 0f) {
				for (int d = 0; d < Hd; d++)
					gradK[kBase + d] += gsPos * st.qPostRope()[qBase + d];
			}
		}

		// NeoX RoPE adjoint (post-RoPE → pre-RoPE)
		Phi3Rope.ropeExtBackward(gradQ, pos, NH, Hd, ropeCfg);
		Phi3Rope.ropeExtBackward(gradK, pos, NKV, Hd, ropeCfg);

		// LoRA branches accumulate against their own logical slice inputs.
		float[] gradXNorm1 = new float[H];
		addLoraBackward(gradXNorm1, absLayer, "wq", gradQ, st.xNorm1());
		addLoraBackward(gradXNorm1, absLayer, "wk", gradK, st.xNorm1());
		addLoraBackward(gradXNorm1, absLayer, "wv", gradV, st.xNorm1());

		// Frozen-projection adjoint: [gradQ, gradK, gradV] concatenated, one transpose on attnQkv.
		float[] gradQkv = new float[H + 2 * kvDim];
		System.arraycopy(gradQ, 0, gradQkv, 0, H);
		System.arraycopy(gradK, 0, gradQkv, H, kvDim);
		System.arraycopy(gradV, 0, gradQkv, H + kvDim, kvDim);
		addInPlace(gradXNorm1, LoraResidentWeights.transposedMatVec(attnQkv[li],
				attnQkvDev != null ? attnQkvDev[li] : null, gradQkv, H + 2 * kvDim, H));

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

	private static float[] softmaxCopy(float[] logits) {
		float[] out = logits.clone();
		LlamaTransformerHandler.softmax(out, out.length);
		return out;
	}

	private static void addInPlace(float[] dst, float[] src) {
		for (int i = 0; i < dst.length; i++)
			dst[i] += src[i];
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
