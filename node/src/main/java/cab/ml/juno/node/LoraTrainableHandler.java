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
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import cab.ml.juno.lora.DoraMagnitude;
import cab.ml.juno.lora.DoraProjection;
import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraGradients;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraTrainContext;
import cab.ml.juno.lora.QaLoraAdapter;

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
 * LoraAdapterSet adapters = LoraQvInitializer.qv(cfg, rank = 8, alpha = 8f, rng);
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
public final class LoraTrainableHandler implements LoraTrainingHandler {

	private static final Logger log = Logger.getLogger(LoraTrainableHandler.class.getName());

	// ── Activations stored per layer during a training forward ────────────────

	private record LayerState(float[] xIn, // residual stream before this layer [H]
			float[] xNorm1, // after pre-attention rmsNorm [H]
			float[] qPostRope, // Q after RoPE [numHeads*headDim]
			float[][] attnW, // attention weights per head [numHeads][seqLen]
			float[] attnOut, // attention output before wo [H]
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

	/** Optional Qwen2-style Q/K/V biases; null when the GGUF has no bias tensors. */
	private final float[][] bq, bk, bv;

	// ── LoRA adapters ─────────────────────────────────────────────────────────

	private final LoraAdapterSet loraAdapters;
	/** DoRA projections keyed by absolute {@code layer:proj}; empty when unused. */
	private final Map<String, DoraProjection> doraByKey;
	private long doraSeenGeneration = -1;

	/** Active training dropout context; disabled during inference and evaluation. */
	private LoraTrainContext trainCtx = LoraTrainContext.disabled();
	/** Token position for the active training forward/backward step. */
	private int trainTokenPos;

	private final MatVec backend;
	private ResidentWeightMatrix[] wqDev;
	private ResidentWeightMatrix[] wkDev;
	private ResidentWeightMatrix[] wvDev;
	private ResidentWeightMatrix[] woDev;
	private ResidentWeightMatrix[] wGateDev;
	private ResidentWeightMatrix[] wUpDev;
	private ResidentWeightMatrix[] wDownDev;
	private ResidentWeightMatrix outputProjDev;
	/** Tier-9 microbatched GEMM scratch; non-null when resident FP32 weights are uploaded. */
	private GpuBlasOps blasOps;

	/** Nanosecond accumulators for Tier-9 train-step timing subsets (reset per chunk). */
	private long accFrozenForwardNs;
	private long accFrozenTransposeNs;
	private long accAdapterBackwardNs;
	private boolean timingActive;

	// ── Inference KV cache ────────────────────────────────────────────────────

	private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
	private static final int MAX_SEQ_LEN = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64;

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load a model shard and attach LoRA adapters.
	 *
	 * @param modelPath path to the GGUF file
	 * @param context   which layers/embeddings this node is responsible for
	 * @param adapters  LoRA adapters (typically created with {@link LoraQvInitializer#qv})
	 */
	public static LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters)
			throws IOException {
		return load(modelPath, context, adapters, ForwardPassHandlerLoader.selectLoraBackend());
	}

	/**
	 * Load with an explicit {@link MatVec} (matches {@link ForwardPassHandlerLoader#load}
	 * when adapters are supplied for inference-only playback).
	 */
	public static LoraTrainableHandler load(Path modelPath, ShardContext context, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		log.info("Loading LoRA handler: layers " + context.startLayer() + "–" + context.endLayer() + "  adapters="
				+ adapters.size() + "  backend=" + backend.getClass().getSimpleName() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			LoraInitializer.validate(adapters, cfg);
			DoraInitializer.verifyFingerprints(r, adapters);
			return new LoraTrainableHandler(r, cfg, context, adapters, backend);
		}
	}

	private LoraTrainableHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, LoraAdapterSet adapters,
			MatVec backend) throws IOException {
		this.cfg = cfg;
		this.loraAdapters = adapters;
		this.backend = backend;
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

		float[][] bqLocal = null;
		float[][] bkLocal = null;
		float[][] bvLocal = null;
		if (L > 0 && r.hasTensor("blk." + startLayer + ".attn_q.bias")) {
			bqLocal = new float[L][];
			bkLocal = new float[L][];
			bvLocal = new float[L][];
		}

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
			if (bqLocal != null) {
				bqLocal[li] = r.tensor("blk." + i + ".attn_q.bias");
				bkLocal[li] = r.tensor("blk." + i + ".attn_k.bias");
				bvLocal[li] = r.tensor("blk." + i + ".attn_v.bias");
			}
		}
		this.bq = bqLocal;
		this.bk = bkLocal;
		this.bv = bvLocal;
		if (bqLocal != null)
			log.info("LoRA handler: loaded QKV biases (Qwen2-style)");

		wqDev = wkDev = wvDev = woDev = wGateDev = wUpDev = wDownDev = null;
		outputProjDev = null;
		if (backend instanceof GpuMatVec gpu) {
			uploadResidentWeights(gpu, L);
		}

		this.doraByKey = buildDoraProjections(r, adapters, startLayer, endLayer);
		this.doraSeenGeneration = adapters.doraGeneration();
		if (!doraByKey.isEmpty()) {
			long t0 = System.currentTimeMillis();
			for (DoraProjection d : doraByKey.values())
				d.refresh();
			LoraMetricsIdentity identity = LoraMetricsIdentity.fromAdapterSet(adapters, cfg.architecture(),
					LoraMetricsIdentity.resolveTrainDevice(true));
			LoraNormRefreshEvent nr = new LoraNormRefreshEvent();
			nr.begin();
			identity.apply(nr);
			nr.projectionCount = doraByKey.size();
			nr.layerCount = endLayer - startLayer;
			nr.durationMs = System.currentTimeMillis() - t0;
			nr.reason = "load";
			nr.commit();
		}
	}

	private void uploadResidentWeights(GpuMatVec gpu, int L) {
		boolean microbatch = LoraResidentWeights.microbatchSize() > 1;
		boolean half = !microbatch && gpu.supportsHalfResident();
		log.info("LoRA handler: uploading projection weights to GPU ("
				+ (half ? "FP16" : "FP32")
				+ (microbatch ? ", microbatch=" + LoraResidentWeights.microbatchSize() : "")
				+ ")…");
		int H = cfg.hiddenDim();
		int KV = cfg.kvDim();
		int I = cfg.intermediateSize();
		int V = cfg.vocabSize();
		ResidentWeightMatrix[] wqD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] wkD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] wvD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] woD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] wGateD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] wUpD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] wDownD = new ResidentWeightMatrix[L];
		ResidentWeightMatrix[] outHolder = new ResidentWeightMatrix[1];
		GpuBlasOps ops = null;
		try {
			for (int li = 0; li < L; li++) {
				wqD[li] = LoraResidentWeights.uploadQuant(gpu, wq[li], H, H);
				wkD[li] = LoraResidentWeights.uploadQuant(gpu, wk[li], KV, H);
				wvD[li] = LoraResidentWeights.uploadQuant(gpu, wv[li], KV, H);
				woD[li] = LoraResidentWeights.uploadQuant(gpu, wo[li], H, H);
				wGateD[li] = LoraResidentWeights.uploadQuant(gpu, wGate[li], I, H);
				wUpD[li] = LoraResidentWeights.uploadQuant(gpu, wUp[li], I, H);
				wDownD[li] = LoraResidentWeights.uploadQuant(gpu, wDown[li], H, I);
			}
			if (outputProj != null)
				outHolder[0] = LoraResidentWeights.uploadQuant(gpu, outputProj, V, H);
			if (microbatch)
				ops = GpuBlasOps.of(gpu);
			this.wqDev = wqD;
			this.wkDev = wkD;
			this.wvDev = wvD;
			this.woDev = woD;
			this.wGateDev = wGateD;
			this.wUpDev = wUpD;
			this.wDownDev = wDownD;
			this.outputProjDev = outHolder[0];
			this.blasOps = ops;
			log.info("LoRA handler: GPU weight upload complete (" + (half ? "FP16" : "FP32") + ").");
		} catch (IllegalStateException ex) {
			if (ops != null)
				ops.close();
			LoraResidentWeights.tryRecoverFromUploadOom(ex, log, () -> {
				LoraResidentWeights.closeArray(wqD);
				LoraResidentWeights.closeArray(wkD);
				LoraResidentWeights.closeArray(wvD);
				LoraResidentWeights.closeArray(woD);
				LoraResidentWeights.closeArray(wGateD);
				LoraResidentWeights.closeArray(wUpD);
				LoraResidentWeights.closeArray(wDownD);
				LoraResidentWeights.closeQuietly(outHolder[0]);
			});
		}
	}

	private float[] matVecLayer(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, float[] x, int rows,
			int cols) {
		if (!timingActive)
			return LoraResidentWeights.matVec(quant, dev, x, rows, cols);
		long t0 = System.nanoTime();
		float[] y = LoraResidentWeights.matVec(quant, dev, x, rows, cols);
		accFrozenForwardNs += System.nanoTime() - t0;
		return y;
	}

	private float[][] matVecBatchLayer(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, float[][] X,
			int batch, int rows, int cols) {
		if (!timingActive)
			return LoraResidentWeights.matVecBatch(quant, dev, blasOps, X, batch, rows, cols);
		long t0 = System.nanoTime();
		float[][] y = LoraResidentWeights.matVecBatch(quant, dev, blasOps, X, batch, rows, cols);
		accFrozenForwardNs += System.nanoTime() - t0;
		return y;
	}

	/**
	 * Frozen transpose {@code W^T * g}: uses resident GPU matrices when uploaded,
	 * otherwise the quantized CPU path.
	 */
	private float[] transposedMatVecLayer(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, float[] g,
			int rows, int cols) {
		if (!timingActive)
			return LoraResidentWeights.transposedMatVec(quant, dev, g, rows, cols);
		long t0 = System.nanoTime();
		float[] y = LoraResidentWeights.transposedMatVec(quant, dev, g, rows, cols);
		accFrozenTransposeNs += System.nanoTime() - t0;
		return y;
	}

	private float[][] transposedMatVecBatchLayer(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev,
			float[][] G, int batch, int rows, int cols) {
		if (!timingActive)
			return LoraResidentWeights.transposedMatVecBatch(quant, dev, blasOps, G, batch, rows, cols);
		long t0 = System.nanoTime();
		float[][] y = LoraResidentWeights.transposedMatVecBatch(quant, dev, blasOps, G, batch, rows, cols);
		accFrozenTransposeNs += System.nanoTime() - t0;
		return y;
	}

	private void resetStepTiming() {
		accFrozenForwardNs = 0L;
		accFrozenTransposeNs = 0L;
		accAdapterBackwardNs = 0L;
	}

	private LoraStepTiming finishStepTiming(long forwardMs, long backwardMs) {
		LoraStepTiming t = new LoraStepTiming();
		t.frozenForwardMs = nsToMs(accFrozenForwardNs);
		t.frozenTransposeBackwardMs = nsToMs(accFrozenTransposeNs);
		t.adapterBackwardMs = nsToMs(accAdapterBackwardNs);
		long accounted = t.frozenForwardMs + t.frozenTransposeBackwardMs + t.adapterBackwardMs;
		t.attentionNonlinearMs = Math.max(0L, forwardMs + backwardMs - accounted);
		t.transferMs = 0L; // populated when H2D/D2H counters exist
		return t;
	}

	private static long nsToMs(long ns) {
		return ns / 1_000_000L;
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

		evt.handlerType = "lora";
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
		if (blasOps != null) {
			blasOps.close();
			blasOps = null;
		}
		LoraResidentWeights.closeArray(wqDev);
		LoraResidentWeights.closeArray(wkDev);
		LoraResidentWeights.closeArray(wvDev);
		LoraResidentWeights.closeArray(woDev);
		LoraResidentWeights.closeArray(wGateDev);
		LoraResidentWeights.closeArray(wUpDev);
		LoraResidentWeights.closeArray(wDownDev);
		LoraResidentWeights.closeQuietly(outputProjDev);
		wqDev = wkDev = wvDev = woDev = wGateDev = wUpDev = wDownDev = null;
		outputProjDev = null;
	}

	// ── Training step ─────────────────────────────────────────────────────────

	/**
	 * Forward and backward over {@code tokens}, accumulating <em>unnormalized</em>
	 * summed gradients into each adapter. Does not clear gradients or step the
	 * optimizer. Every prediction position contributes to the loss.
	 *
	 * @param tokens input token sequence, length ≥ 2
	 * @return summed loss and prediction count for token-weighted aggregation
	 */
	@Override
	public LoraGradientResult computeGradients(int[] tokens) {
		return computeGradients(tokens, null, LoraTrainContext.disabled());
	}

	/**
	 * Like {@link #computeGradients(int[])}, but only positions where
	 * {@code lossMask[pos]} is true contribute loss/gradients for predicting
	 * {@code tokens[pos + 1]}. Pass {@code null} to train every position.
	 *
	 * <p>
	 * Completion-only masks (assistant answer tokens, not the user prompt) are
	 * required for {@code /train-qa}: otherwise LoRA overfits the answer token
	 * and can reply with it for every prompt.
	 *
	 * @param tokens   input token sequence, length ≥ 2
	 * @param lossMask length {@code tokens.length - 1}, or {@code null} for all-true
	 */
	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask) {
		return computeGradients(tokens, lossMask, LoraTrainContext.disabled());
	}

	/**
	 * Gradient computation with optional deterministic train-only dropout context.
	 */
	@Override
	public LoraGradientResult computeGradients(int[] tokens, boolean[] lossMask, LoraTrainContext ctx) {
		if (tokens.length < 2)
			throw new IllegalArgumentException("tokens.length must be >= 2 (need at least one prediction pair)");
		int T = tokens.length - 1;
		if (lossMask != null && lossMask.length != T)
			throw new IllegalArgumentException(
					"lossMask.length must equal tokens.length - 1 (got " + lossMask.length + " vs " + T + ")");

		trainCtx = ctx != null ? ctx : LoraTrainContext.disabled();
		resetStepTiming();
		timingActive = true;
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
			float[][] xCur = new float[T][];
			for (int pos = 0; pos < T; pos++)
				xCur[pos] = embedding(tokens[pos]);

			// Per-layer microbatched frozen linears across positions (Tier 9).
			for (int li = 0; li < L; li++) {
				int I = cfg.intermediateSize();
				int Hd = cfg.headDim();
				int NH = cfg.numHeads();
				int mb = LoraResidentWeights.microbatchSize();
				for (int start = 0; start < T; start += mb) {
					int n = Math.min(mb, T - start);
					float[][] xNorm1 = new float[n][];
					for (int i = 0; i < n; i++) {
						trainTokenPos = start + i;
						xNorm1[i] = LlamaTransformerHandler.rmsNorm(xCur[start + i], attnNorm[li], cfg.rmsNormEps());
					}
					float[][] qB = matVecBatchLayer(wq[li], wqDev != null ? wqDev[li] : null, xNorm1, n, H, H);
					float[][] kB = matVecBatchLayer(wk[li], wkDev != null ? wkDev[li] : null, xNorm1, n, kvDim, H);
					float[][] vB = matVecBatchLayer(wv[li], wvDev != null ? wvDev[li] : null, xNorm1, n, kvDim, H);

					float[][] attnOutB = new float[n][];
					float[][][] attnWB = new float[n][][];
					float[][] qPostRopeB = new float[n][];
					for (int i = 0; i < n; i++) {
						int pos = start + i;
						trainTokenPos = pos;
						float[] q = qB[i];
						float[] k = kB[i];
						float[] v = vB[i];
						applyLoraInPlace(q, li, "wq", xNorm1[i]);
						applyLoraInPlace(k, li, "wk", xNorm1[i]);
						applyLoraInPlace(v, li, "wv", xNorm1[i]);
						addBiasInPlace(q, bq, li);
						addBiasInPlace(k, bk, li);
						addBiasInPlace(v, bv, li);
						LlamaTransformerHandler.rope(q, pos, cfg.numHeads(), Hd, cfg.ropeTheta());
						LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), Hd, cfg.ropeTheta());
						qPostRopeB[i] = q.clone();
						System.arraycopy(k, 0, kCache[li], pos * kvDim, kvDim);
						System.arraycopy(v, 0, vCache[li], pos * kvDim, kvDim);

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
									dot += q[qBase + d] * kCache[li][kOff + d];
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
									attnOut[outBase + d] += w * vCache[li][vOff + d];
							}
						}
						attnOutB[i] = attnOut;
						attnWB[i] = attnW;
					}

					float[][] attnProjB = matVecBatchLayer(wo[li], woDev != null ? woDev[li] : null, attnOutB, n, H, H);
					float[][] xRes2B = new float[n][];
					float[][] xNorm2B = new float[n][];
					for (int i = 0; i < n; i++) {
						trainTokenPos = start + i;
						applyLoraInPlace(attnProjB[i], li, "wo", attnOutB[i]);
						xRes2B[i] = LlamaTransformerHandler.add(xCur[start + i], attnProjB[i]);
						xNorm2B[i] = LlamaTransformerHandler.rmsNorm(xRes2B[i], ffnNorm[li], cfg.rmsNormEps());
					}
					float[][] gateB = matVecBatchLayer(wGate[li], wGateDev != null ? wGateDev[li] : null, xNorm2B, n, I, H);
					float[][] upB = matVecBatchLayer(wUp[li], wUpDev != null ? wUpDev[li] : null, xNorm2B, n, I, H);
					float[][] hiddenB = new float[n][];
					for (int i = 0; i < n; i++) {
						trainTokenPos = start + i;
						applyLoraInPlace(gateB[i], li, "wgate", xNorm2B[i]);
						applyLoraInPlace(upB[i], li, "wup", xNorm2B[i]);
						float[] hidden = new float[I];
						for (int j = 0; j < I; j++)
							hidden[j] = LlamaTransformerHandler.silu(gateB[i][j]) * upB[i][j];
						hiddenB[i] = hidden;
					}
					float[][] ffnOutB = matVecBatchLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, hiddenB, n, H, I);
					for (int i = 0; i < n; i++) {
						int pos = start + i;
						trainTokenPos = pos;
						applyLoraInPlace(ffnOutB[i], li, "wdown", hiddenB[i]);
						allStates[pos][li] = new LayerState(xCur[pos].clone(), xNorm1[i], qPostRopeB[i], attnWB[i],
								attnOutB[i], xRes2B[i], xNorm2B[i], gateB[i], upB[i], hiddenB[i]);
						xCur[pos] = LlamaTransformerHandler.add(xRes2B[i], ffnOutB[i]);
					}
				}
			}

			if (hasOutputProj) {
				for (int pos = 0; pos < T; pos++) {
					allXFinal[pos] = xCur[pos].clone();
					allXNormFinal[pos] = LlamaTransformerHandler.rmsNorm(xCur[pos], outputNorm, cfg.rmsNormEps());
				}
				float[][] logitsB = matVecBatchLayer(outputProj, outputProjDev, allXNormFinal, T, cfg.vocabSize(), H);
				for (int pos = 0; pos < T; pos++)
					allProbs[pos] = softmaxCopy(logitsB[pos]);
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
			// Collect loss positions for microbatched output transpose.
			int[] lossPos = new int[predictionCount];
			int lp = 0;
			for (int pos = 0; pos < T; pos++) {
				if (lossMask != null && !lossMask[pos])
					continue;
				lossPos[lp++] = pos;
			}
			float[][] gradXByPos = new float[T][];
			if (hasOutputProj && predictionCount > 0) {
				float[][] gradLogitsB = new float[predictionCount][];
				for (int i = 0; i < predictionCount; i++) {
					int pos = lossPos[i];
					float[] gradLogits = allProbs[pos].clone();
					gradLogits[tokens[pos + 1]] -= 1.0f;
					gradLogitsB[i] = gradLogits;
				}
				float[][] gradXNormB = transposedMatVecBatchLayer(outputProj, outputProjDev, gradLogitsB,
						predictionCount, cfg.vocabSize(), H);
				for (int i = 0; i < predictionCount; i++) {
					int pos = lossPos[i];
					gradXByPos[pos] = rmsNormBackward(allXFinal[pos], outputNorm, gradXNormB[i], cfg.rmsNormEps());
				}
			}
			for (int i = 0; i < predictionCount; i++) {
				int pos = lossPos[i];
				trainTokenPos = pos;
				float[] gradX = hasOutputProj ? gradXByPos[pos] : new float[H];
				for (int li = L - 1; li >= 0; li--) {
					gradX = backwardLayer(gradX, li, pos, allStates[pos][li], kCache[li], vCache[li], pos + 1);
				}
			}
			long backwardMs = System.currentTimeMillis() - t1;

			return new LoraGradientResult(lossSum, predictionCount, forwardMs, backwardMs, finishStepTiming(forwardMs, backwardMs));
		} finally {
			timingActive = false;
			trainCtx = LoraTrainContext.disabled();
		}
	}

	/**
	 * Forward-only teacher-forced loss with local K/V buffers. No activation
	 * retention for backward, no gradient accumulation, no optimizer mutation, no
	 * dropout, and no pollution of persistent inference caches.
	 *
	 * @return token-weighted loss sum and prediction count
	 */
	@Override
	public LoraGradientResult evaluateLoss(int[] tokens) {
		return evaluateLoss(tokens, null);
	}

	@Override
	public LoraGradientResult evaluateLoss(int[] tokens, boolean[] lossMask) {
		if (tokens.length < 2)
			throw new IllegalArgumentException("tokens.length must be >= 2 (need at least one prediction pair)");
		int T = tokens.length - 1;
		if (lossMask != null && lossMask.length != T)
			throw new IllegalArgumentException(
					"lossMask.length must equal tokens.length - 1 (got " + lossMask.length + " vs " + T + ")");

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
				float[] logits = matVecLayer(outputProj, outputProjDev, xn, cfg.vocabSize(), H);
				float[] probs = softmaxCopy(logits);
				lossSum -= (float) Math.log(Math.max(probs[tokens[pos + 1]], 1e-9f));
			}
			predictionCount++;
		}
		long forwardMs = System.currentTimeMillis() - t0;
		return new LoraGradientResult(lossSum, predictionCount, forwardMs, 0L);
	}

	/**
	 * One teacher-forcing training step over a token sequence (legacy one-chunk
	 * update).
	 *
	 * <p>
	 * Clears gradients, computes one sequence, normalizes by prediction count,
	 * steps the optimizer once, and returns mean loss. Clipping is disabled
	 * ({@code maxNorm = 0}) for numerical compatibility with pre-Tier-1 callers.
	 *
	 * @param tokens    input token sequence, length ≥ 2
	 * @param optimizer Adam optimizer to apply after backward
	 * @return mean cross-entropy loss (nats) for this sequence
	 */
	public float trainStep(int[] tokens, LoraAdamOptimizer optimizer) {
		loraAdapters.zeroAllGrads();

		LoraMetricsIdentity identity = LoraMetricsIdentity.fromAdapterSet(loraAdapters, cfg.architecture(),
				LoraMetricsIdentity.resolveTrainDevice(true));
		LoraTrainEvent event = new LoraTrainEvent();
		event.begin();
		identity.apply(event);
		event.step = optimizer.step() + 1;
		event.numTokens = tokens.length;
		event.chunkCount = 1;

		LoraGradientResult r = computeGradients(tokens);
		event.forwardMs = r.forwardMs();
		event.backwardMs = r.backwardMs();
		r.timing().apply(event);
		event.predictionCount = r.predictionCount();

		LoraGradients.PrepResult prep = LoraGradients.prepare(loraAdapters, r.predictionCount(), 0f);
		event.globalGradNorm = (float) prep.globalNorm();
		event.clipScale = prep.scale();
		event.clipped = prep.clipped();

		long t2 = System.currentTimeMillis();
		optimizer.step(loraAdapters);
		event.optimizerMs = System.currentTimeMillis() - t2;

		float meanLoss = r.lossSum() / r.predictionCount();
		event.loss = meanLoss;
		event.totalMs = event.forwardMs + event.backwardMs + event.optimizerMs;
		event.commit();

		if (!doraByKey.isEmpty()) {
			LoraNormRefreshEvent nr = new LoraNormRefreshEvent();
			nr.begin();
			identity.apply(nr);
			nr.projectionCount = doraByKey.size();
			nr.layerCount = endLayer - startLayer;
			nr.durationMs = 0;
			nr.reason = "post-step";
			nr.commit();
		}

		return meanLoss;
	}

	/** Expose adapters for orchestration (accumulation / clipping outside the handler). */
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
		return LoraModelLayout.forArchitecture(cfg.architecture(), cfg);
	}

	public LlamaConfig config() {
		return cfg;
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
		kvCacheK.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		kvCacheV.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
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

		float[] q = matVecLayer(wq[li], wqDev != null ? wqDev[li] : null, xNorm1, H, H);
		float[] k = matVecLayer(wk[li], wkDev != null ? wkDev[li] : null, xNorm1, kvDim, H);
		float[] v = matVecLayer(wv[li], wvDev != null ? wvDev[li] : null, xNorm1, kvDim, H);

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);
		addBiasInPlace(q, bq, li);
		addBiasInPlace(k, bk, li);
		addBiasInPlace(v, bv, li);

		LlamaTransformerHandler.rope(q, pos, cfg.numHeads(), cfg.headDim(), cfg.ropeTheta());
		LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = matVecLayer(wo[li], woDev != null ? woDev[li] : null, attnOut, H, H);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/** Frozen QKV bias add (Qwen2); no-op when biases are absent. */
	private static void addBiasInPlace(float[] x, float[][] biases, int li) {
		if (biases == null)
			return;
		float[] b = biases[li];
		for (int i = 0; i < x.length; i++)
			x[i] += b[i];
	}

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
		refreshDoraIfNeeded();
		float[] delta;
		if (trainCtx.dropoutEnabled()) {
			int projOrd = LoraProjection.fromKey(proj).ordinal();
			delta = lora.forwardTrain(input, trainCtx.dropoutRate(), trainCtx.rootSeed(), trainCtx.optimizerUpdate(),
					trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
		} else {
			delta = lora.forward(input);
		}
		DoraProjection dora = doraByKey.get(absLayer + ":" + proj);
		if (dora == null) {
			for (int i = 0; i < out.length; i++)
				out[i] += delta[i];
			return;
		}
		float[] direction = new float[out.length];
		for (int i = 0; i < out.length; i++)
			direction[i] = out[i] + delta[i];
		float[] scaled = dora.scaleDirectionOutput(direction);
		System.arraycopy(scaled, 0, out, 0, out.length);
	}

	/**
	 * Scale an outgoing projection gradient for DoRA (accumulates magnitude grads)
	 * or return {@code gradOut} unchanged for plain LoRA.
	 */
	private float[] maybeScaleDoraGrad(int absLayer, String proj, float[] gradOut) {
		DoraProjection dora = doraByKey.get(absLayer + ":" + proj);
		if (dora == null)
			return gradOut;
		refreshDoraIfNeeded();
		return dora.scaleGradient(gradOut);
	}

	private float[] loraBackward(int absLayer, String proj, float[] gradDelta, float[] input) {
		long t0 = timingActive ? System.nanoTime() : 0L;
		float[] result;
		QaLoraAdapter qa = loraAdapters.getQa(absLayer, proj);
		if (qa != null) {
			if (!trainCtx.dropoutEnabled())
				result = qa.backward(gradDelta, input);
			else {
				int projOrd = LoraProjection.fromKey(proj).ordinal();
				result = qa.backwardTrain(gradDelta, input, trainCtx.dropoutRate(), trainCtx.rootSeed(),
						trainCtx.optimizerUpdate(), trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
			}
		} else {
			LoraAdapter lora = loraAdapters.get(absLayer, proj);
			if (lora == null)
				return null;
			if (!trainCtx.dropoutEnabled())
				result = lora.backward(gradDelta, input);
			else {
				int projOrd = LoraProjection.fromKey(proj).ordinal();
				result = lora.backwardTrain(gradDelta, input, trainCtx.dropoutRate(), trainCtx.rootSeed(),
						trainCtx.optimizerUpdate(), trainCtx.chunkOrdinal(), trainTokenPos, absLayer, projOrd);
			}
		}
		if (timingActive)
			accAdapterBackwardNs += System.nanoTime() - t0;
		return result;
	}

	private void addLoraBackward(float[] dest, int absLayer, String proj, float[] gradDelta, float[] input) {
		float[] g = loraBackward(absLayer, proj, gradDelta, input);
		if (g != null)
			addInPlace(dest, g);
	}

	private void refreshDoraIfNeeded() {
		if (doraByKey.isEmpty())
			return;
		long gen = loraAdapters.doraGeneration();
		if (gen == doraSeenGeneration)
			return;
		for (DoraProjection d : doraByKey.values())
			d.markDirty();
		doraSeenGeneration = gen;
	}

	private static Map<String, DoraProjection> buildDoraProjections(GgufReader r, LoraAdapterSet adapters, int startLayer,
			int endLayer) throws IOException {
		Map<String, DoraProjection> map = new java.util.HashMap<>();
		for (var entry : adapters.asMap().entrySet()) {
			LoraAdapter a = entry.getValue();
			if (a.mode != LoraMode.DORA)
				continue;
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			if (layer < startLayer || layer >= endLayer)
				continue;
			String projKey = LoraAdapterSet.keyProj(key);
			LoraProjection proj = LoraProjection.fromKey(projKey);
			DoraMagnitude mag = adapters.getMagnitude(layer, projKey);
			if (mag == null)
				throw new IllegalArgumentException("DoRA adapter missing magnitude: " + key);
			float[] w = r.tensor(proj.ggufTensorName(layer));
			map.put(key, new DoraProjection(w, a, mag));
		}
		return map;
	}

	private float[] outputProjection(float[] x) {
		float[] xn = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		return matVecLayer(outputProj, outputProjDev, xn, cfg.vocabSize(), cfg.hiddenDim());
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

		float[] q = matVecLayer(wq[li], wqDev != null ? wqDev[li] : null, xNorm1, H, H);
		float[] k = matVecLayer(wk[li], wkDev != null ? wkDev[li] : null, xNorm1, kvDim, H);
		float[] v = matVecLayer(wv[li], wvDev != null ? wvDev[li] : null, xNorm1, kvDim, H);

		applyLoraInPlace(q, li, "wq", xNorm1);
		applyLoraInPlace(k, li, "wk", xNorm1);
		applyLoraInPlace(v, li, "wv", xNorm1);
		addBiasInPlace(q, bq, li);
		addBiasInPlace(k, bk, li);
		addBiasInPlace(v, bv, li);

		LlamaTransformerHandler.rope(q, pos, cfg.numHeads(), Hd, cfg.ropeTheta());
		LlamaTransformerHandler.rope(k, pos, cfg.numKvHeads(), Hd, cfg.ropeTheta());

		float[] qPostRope = q.clone();

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

		float[] attnProj = matVecLayer(wo[li], woDev != null ? woDev[li] : null, attnOut, H, H);
		applyLoraInPlace(attnProj, li, "wo", attnOut);
		float[] xRes2 = LlamaTransformerHandler.add(x, attnProj);
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(xRes2, ffnNorm[li], cfg.rmsNormEps());

		int I = cfg.intermediateSize();
		float[] gate = matVecLayer(wGate[li], wGateDev != null ? wGateDev[li] : null, xNorm2, I, H);
		float[] up = matVecLayer(wUp[li], wUpDev != null ? wUpDev[li] : null, xNorm2, I, H);
		applyLoraInPlace(gate, li, "wgate", xNorm2);
		applyLoraInPlace(up, li, "wup", xNorm2);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];

		return new LayerState(x.clone(), xNorm1, qPostRope, attnW, attnOut, xRes2, xNorm2, gate, up, hidden);
	}

	/** Compute the layer output from stored state (completes forwardLayerStore). */
	private float[] computeLayerOutput(LayerState st, int li, float[] xIn) {
		int H = cfg.hiddenDim();
		float[] ffnOut = matVecLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, st.hiddenAct(), H,
				cfg.intermediateSize());
		applyLoraInPlace(ffnOut, li, "wdown", st.hiddenAct());
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
		int NKV = cfg.numKvHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();
		int absLayer = li + startLayer;

		// ── FFN residual: xOut = xRes2 + ffnOut ──────────────────────────────
		float[] gradXRes2 = gradOut.clone(); // through residual
		float[] gradFfnOut = gradOut; // to FFN

		// Backward through wDown: ffnOut = wDown × hiddenAct + lora_down(hiddenAct)
		float[] gradFfnScaled = maybeScaleDoraGrad(absLayer, "wdown", gradFfnOut);
		float[] gradHidden = transposedMatVecLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, gradFfnScaled, H,
				I);
		addLoraBackward(gradHidden, absLayer, "wdown", gradFfnScaled, st.hiddenAct());

		// Backward through SwiGLU: hiddenAct[i] = silu(gate[i]) * up[i]
		float[] gradGate = new float[I];
		float[] gradUp = new float[I];
		for (int i = 0; i < I; i++) {
			float g = st.gate()[i];
			float sig = 1f / (1f + (float) Math.exp(-g));
			gradGate[i] = gradHidden[i] * st.up()[i] * sig * (1f + g * (1f - sig));
			gradUp[i] = gradHidden[i] * LlamaTransformerHandler.silu(g);
		}

		// Backward through wGate and wUp (+ LoRA)
		float[] gradGateScaled = maybeScaleDoraGrad(absLayer, "wgate", gradGate);
		float[] gradUpScaled = maybeScaleDoraGrad(absLayer, "wup", gradUp);
		float[] gradXNorm2 = add(
				transposedMatVecLayer(wGate[li], wGateDev != null ? wGateDev[li] : null, gradGateScaled, I, H),
				transposedMatVecLayer(wUp[li], wUpDev != null ? wUpDev[li] : null, gradUpScaled, I, H));
		addLoraBackward(gradXNorm2, absLayer, "wgate", gradGateScaled, st.xNorm2());
		addLoraBackward(gradXNorm2, absLayer, "wup", gradUpScaled, st.xNorm2());

		// Backward through rmsNorm2
		addInPlace(gradXRes2, rmsNormBackward(st.xRes2(), ffnNorm[li], gradXNorm2, cfg.rmsNormEps()));

		// ── Attention residual: xRes2 = xIn + attnProj ───────────────────────
		float[] gradXIn = gradXRes2.clone(); // through residual
		float[] gradAttnProj = gradXRes2;

		// Backward through wo: attnProj = wo × attnOut + lora_o(attnOut)
		float[] gradAttnScaled = maybeScaleDoraGrad(absLayer, "wo", gradAttnProj);
		float[] gradAttnOut = transposedMatVecLayer(wo[li], woDev != null ? woDev[li] : null, gradAttnScaled, H, H);
		addLoraBackward(gradAttnOut, absLayer, "wo", gradAttnScaled, st.attnOut());

		// ── Attention backward ────────────────────────────────────────────────
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] gradQ = new float[NH * Hd];
		float[] gradK = new float[kvDim]; // current position only (truncated BPTT)
		float[] gradV = new float[kvDim];

		for (int h = 0; h < NH; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;
			float[] aw = st.attnW()[h]; // [seqLen]

			float[] gradAttnOut_h = Arrays.copyOfRange(gradAttnOut, qBase, qBase + Hd);

			// 1. gradV at current position
			for (int d = 0; d < Hd; d++)
				gradV[kBase + d] += aw[pos] * gradAttnOut_h[d];

			// 2. Softmax backward through attention weights
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

			// 3. gradQ[h] = scale * Σ_t gradScores[t] * K[t]
			for (int t = 0; t < seqLen; t++) {
				if (gradScores[t] == 0f)
					continue;
				int kOff = t * kvDim + kBase;
				float gs = gradScores[t] * scale;
				for (int d = 0; d < Hd; d++)
					gradQ[qBase + d] += gs * kCacheLayer[kOff + d];
			}

			// 4. gradK at current position: scale * gradScores[pos] * Q[pos]
			float gsPos = gradScores[pos] * scale;
			if (gsPos != 0f) {
				for (int d = 0; d < Hd; d++)
					gradK[kBase + d] += gsPos * st.qPostRope()[qBase + d];
			}
		}

		// Inverse RoPE on Q and K gradients (post-RoPE → pre-RoPE)
		ropeBackward(gradQ, pos, NH, Hd, cfg.ropeTheta());
		ropeBackward(gradK, pos, NKV, Hd, cfg.ropeTheta());

		// ── LoRA / frozen projection backward into xNorm1 ─────────────────────
		float[] gradQScaled = maybeScaleDoraGrad(absLayer, "wq", gradQ);
		float[] gradXNorm1 = transposedMatVecLayer(wq[li], wqDev != null ? wqDev[li] : null, gradQScaled, H, H);
		addLoraBackward(gradXNorm1, absLayer, "wq", gradQScaled, st.xNorm1());

		float[] gradKScaled = maybeScaleDoraGrad(absLayer, "wk", gradK);
		float[] gradXNorm1_k = transposedMatVecLayer(wk[li], wkDev != null ? wkDev[li] : null, gradKScaled, kvDim, H);
		addLoraBackward(gradXNorm1_k, absLayer, "wk", gradKScaled, st.xNorm1());
		addInPlace(gradXNorm1, gradXNorm1_k);

		float[] gradVScaled = maybeScaleDoraGrad(absLayer, "wv", gradV);
		float[] gradXNorm1_v = transposedMatVecLayer(wv[li], wvDev != null ? wvDev[li] : null, gradVScaled, kvDim, H);
		addLoraBackward(gradXNorm1_v, absLayer, "wv", gradVScaled, st.xNorm1());
		addInPlace(gradXNorm1, gradXNorm1_v);

		// Backward through rmsNorm1
		addInPlace(gradXIn, rmsNormBackward(st.xIn(), attnNorm[li], gradXNorm1, cfg.rmsNormEps()));

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
		float[] gate = matVecLayer(wGate[li], wGateDev != null ? wGateDev[li] : null, x, I, H);
		float[] up = matVecLayer(wUp[li], wUpDev != null ? wUpDev[li] : null, x, I, H);
		applyLoraInPlace(gate, li, "wgate", x);
		applyLoraInPlace(up, li, "wup", x);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		float[] down = matVecLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, hidden, H, I);
		applyLoraInPlace(down, li, "wdown", hidden);
		return down;
	}

	/**
	 * RMSNorm backward — delegates to {@link LoraTrainingMath}.
	 */
	static float[] rmsNormBackward(float[] x, float[] w, float[] gradOut, float eps) {
		return LoraTrainingMath.rmsNormBackward(x, w, gradOut, eps);
	}

	/**
	 * Inverse LLaMA adjacent-pair RoPE — delegates to {@link LoraTrainingMath}.
	 */
	static void ropeBackward(float[] g, int pos, int nHeads, int headDim, float ropeTheta) {
		LoraTrainingMath.ropeBackward(g, pos, nHeads, headDim, ropeTheta);
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