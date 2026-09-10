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
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.DenseKvTensor;

/**
 * Qwen3-MoE transformer forward pass — same Q/K-norm attention as
 * {@link Qwen3TransformerHandler} with routed expert SwiGLU FFN.
 */
public final class Qwen3MoeTransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Qwen3MoeTransformerHandler.class.getName());
	private final Qwen3Config cfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	private final float[] tokenEmbd;
	private final float[] outputNorm;
	private final float[] outputProj;

	private final float[][] attnNorm;
	private final float[][] qNorm;
	private final float[][] kNorm;
	private final float[][] ffnNorm;

	private final GgufReader.QuantizedTensor[] attnQ;
	private final GgufReader.QuantizedTensor[] attnK;
	private final GgufReader.QuantizedTensor[] attnV;
	private final GgufReader.QuantizedTensor[] wo;
	private final GgufReader.QuantizedTensor[] ffnGateInp;
	private final GgufReader.QuantizedTensor[] ffnGateExps;
	private final GgufReader.QuantizedTensor[] ffnUpExps;
	private final GgufReader.QuantizedTensor[] ffnDownExps;

	private final MatVec backend;
	private final Map<String, DenseKvTensor[]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, DenseKvTensor[]> kvCacheV = new ConcurrentHashMap<>();
	private final CacheTypeOptions cacheTypes;
	private volatile NodeKVCacheAdapter kvAdapter;

	public static Qwen3MoeTransformerHandler load(Path modelPath, ShardContext context) throws IOException {
		return load(modelPath, context, CpuMatVec.INSTANCE);
	}

	public static Qwen3MoeTransformerHandler load(Path modelPath, ShardContext context, MatVec backend)
			throws IOException {
		log.info("Loading Qwen3-MoE GGUF shard: layers " + context.startLayer() + "–" + context.endLayer()
				+ "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			Qwen3Config config = Qwen3Config.from(r);
			if (!config.isMoe())
				throw new IOException("Expected qwen3moe architecture with expert_count > 0, got " + config);
			log.info("Model: " + config);
			return new Qwen3MoeTransformerHandler(r, config, context, backend);
		}
	}

	private Qwen3MoeTransformerHandler(GgufReader r, Qwen3Config cfg, ShardContext ctx, MatVec backend)
			throws IOException {
		this.cfg = cfg;
		this.backend = backend;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();
		this.cacheTypes = CacheTypeOptions.fromEnv();
		log.info(cacheTypes.policySummary());

		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int headDim = cfg.headDim();
		int nExp = cfg.expertCount();
		int expFf = cfg.expertFeedForwardLength();

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
		ffnGateInp = new GgufReader.QuantizedTensor[L];
		ffnGateExps = new GgufReader.QuantizedTensor[L];
		ffnUpExps = new GgufReader.QuantizedTensor[L];
		ffnDownExps = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			qNorm[li] = r.tensor("blk." + i + ".attn_q_norm.weight");
			kNorm[li] = r.tensor("blk." + i + ".attn_k_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");

			attnQ[li] = r.tensorRaw("blk." + i + ".attn_q.weight");
			attnK[li] = r.tensorRaw("blk." + i + ".attn_k.weight");
			attnV[li] = r.tensorRaw("blk." + i + ".attn_v.weight");
			wo[li] = r.tensorRaw("blk." + i + ".attn_output.weight");

			ffnGateInp[li] = r.tensorRaw("blk." + i + ".ffn_gate_inp.weight");
			ffnGateExps[li] = r.tensorRaw("blk." + i + ".ffn_gate_exps.weight");
			ffnUpExps[li] = r.tensorRaw("blk." + i + ".ffn_up_exps.weight");
			ffnDownExps[li] = r.tensorRaw("blk." + i + ".ffn_down_exps.weight");

			if (qNorm[li].length != headDim || kNorm[li].length != headDim)
				throw new IOException("Layer " + i + ": q/k norm size mismatch");
		}

		log.info("Qwen3-MoE shard loaded — " + L + " layers, " + nExp + " experts, top-" + cfg.expertUsedCount()
				+ ", expertFf=" + expFf);
	}

	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight"))
			return r.tensor("output.weight");
		return r.tensor("token_embd.weight");
	}

	@Override
	public void releaseGpuResources() {
		// CPU quantised path only for MoE v1
	}

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long start = System.nanoTime();
		ForwardPassEvent evt = new ForwardPassEvent();
		evt.begin();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		ForwardResult result;
		if (hasOutputProj) {
			result = ForwardResult.logits(request.requestId(), outputProjection(x), System.nanoTime() - start);
		} else {
			result = ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
		}

		evt.handlerType = "qwen3moe";
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
	public MultiDecodeForwardResult forwardMultiDecode(MultiDecodeForwardRequest request, ShardContext context) {
		long start = System.nanoTime();
		int N = request.batchSize();
		int H = cfg.hiddenDim();
		var requestIds = request.requestIds();
		int[] positions = request.startPositions();

		float[][] x;
		if (hasEmbeddings && request.isFirstNode()) {
			x = new float[N][H];
			int actualVocab = tokenEmbd.length / H;
			for (int b = 0; b < N; b++) {
				int tokenId = Math.max(0, Math.min(request.tokenIds()[b], actualVocab - 1));
				System.arraycopy(tokenEmbd, tokenId * H, x[b], 0, H);
			}
		} else {
			x = new float[N][H];
			float[] flat = request.activations();
			for (int b = 0; b < N; b++)
				System.arraycopy(flat, b * H, x[b], 0, H);
		}

		x = runLayersMultiDecode(x, requestIds, positions);

		if (hasOutputProj) {
			float[][] logits = outputProjectionBatch(x);
			return new MultiDecodeForwardResult(logits, null, N, System.nanoTime() - start);
		}

		float[] flat = new float[N * H];
		for (int b = 0; b < N; b++)
			System.arraycopy(x[b], 0, flat, b * H, H);
		return new MultiDecodeForwardResult(null, flat, N, System.nanoTime() - start);
	}

	private float[][] runLayersMultiDecode(float[][] x, java.util.List<String> requestIds, int[] positions) {
		int N = x.length;
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();
		int maxPos = 0;
		for (int pos : positions)
			maxPos = Math.max(maxPos, pos);

		DenseKvTensor[][] kCaches = new DenseKvTensor[N][];
		DenseKvTensor[][] vCaches = new DenseKvTensor[N][];

		NodeKVCacheAdapter a = kvAdapter;
		for (int i = 0; i < N; i++) {
			String requestId = requestIds.get(i);
			int pos = positions[i];

			kvCacheK.putIfAbsent(requestId, newKLayers(L));
			kvCacheV.computeIfAbsent(requestId, k -> newVLayers(L));
			DenseKvTensor[] kCache = kvCacheK.get(requestId);
			DenseKvTensor[] vCache = kvCacheV.get(requestId);

			for (int li = 0; li < L; li++) {
				kCache[li].ensureCapacity(pos);
				vCache[li].ensureCapacity(pos);
			}
			kCaches[i] = kCache;
			vCaches[i] = vCache;
		}

		MoeBatchWorkspace ws = new MoeBatchWorkspace(N, cfg.hiddenDim(), cfg.qDim(), cfg.kvDim(),
				cfg.numHeads(), maxPos + 1, cacheTypes.usesQuantized());

		for (int li = 0; li < L; li++) {
			DenseKvTensor[] kLayers = new DenseKvTensor[N];
			DenseKvTensor[] vLayers = new DenseKvTensor[N];
			for (int i = 0; i < N; i++) {
				kLayers[i] = kCaches[i][li];
				vLayers[i] = vCaches[i][li];
			}
			x = transformerLayerMultiDecode(x, li, positions, kLayers, vLayers, ws);
		}

		if (a != null) {
			for (int i = 0; i < N; i++) {
				int seqLen = positions[i] + 1;
				for (int li = 0; li < L; li++)
					a.flush(requestIds.get(i), startLayer + li, kCaches[i][li], vCaches[i][li], seqLen);
			}
		}
		return x;
	}

	private static final class MoeBatchWorkspace {
		final float[][] norm1, norm2, q, k, v, attnOut, attnProj;
		final float[] scores;
		final float[] kDequant, vDequant;

		MoeBatchWorkspace(int W, int H, int qDim, int kvDim, int numHeads, int maxSeqLen, boolean quantKv) {
			norm1 = new float[W][H];
			norm2 = new float[W][H];
			q = new float[W][qDim];
			k = new float[W][kvDim];
			v = new float[W][kvDim];
			attnOut = new float[W][qDim];
			attnProj = new float[W][H];
			scores = new float[numHeads * maxSeqLen];
			if (quantKv) {
				kDequant = new float[maxSeqLen * kvDim];
				vDequant = new float[maxSeqLen * kvDim];
			} else {
				kDequant = null;
				vDequant = null;
			}
		}
	}

	private float[][] transformerLayerMultiDecode(float[][] x, int li, int[] positions,
			DenseKvTensor[] kCacheLayers, DenseKvTensor[] vCacheLayers, MoeBatchWorkspace ws) {
		int N = x.length;
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();

		for (int b = 0; b < N; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);

		sgemmLayerInto(attnQ[li], ws.norm1, ws.q, qDim, H);
		sgemmLayerInto(attnK[li], ws.norm1, ws.k, kvDim, H);
		sgemmLayerInto(attnV[li], ws.norm1, ws.v, kvDim, H);

		for (int b = 0; b < N; b++) {
			Qwen3TransformerHandler.rmsNormPerHead(ws.q[b], qNorm[li], cfg.numHeads(), cfg.headDim(),
					cfg.rmsNormEps());
			Qwen3TransformerHandler.rmsNormPerHead(ws.k[b], kNorm[li], cfg.numKvHeads(), cfg.headDim(),
					cfg.rmsNormEps());
		}

		for (int b = 0; b < N; b++) {
			int pos = positions[b];
			Qwen3Rope.apply(ws.q[b], pos, cfg.numHeads(), cfg.headDim(), cfg.rope());
			Qwen3Rope.apply(ws.k[b], pos, cfg.numKvHeads(), cfg.headDim(), cfg.rope());
		}

		for (int b = 0; b < N; b++) {
			int pos = positions[b];
			kCacheLayers[b].writeToken(pos, ws.k[b]);
			vCacheLayers[b].writeToken(pos, ws.v[b]);
		}

		for (int b = 0; b < N; b++) {
			int seqLen = positions[b] + 1;
			float[] kView = kCacheLayers[b].viewForAttention(seqLen, ws.kDequant);
			float[] vView = vCacheLayers[b].viewForAttention(seqLen, ws.vDequant);
			gqaInto(ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
		}

		sgemmLayerInto(wo[li], ws.attnOut, ws.attnProj, H, qDim);

		for (int b = 0; b < N; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.attnProj[b][d];

		for (int b = 0; b < N; b++) {
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);
			float[] ffnOut = moeFfn(ws.norm2[b], li);
			for (int d = 0; d < H; d++)
				x[b][d] += ffnOut[d];
		}

		return x;
	}

	private void sgemmLayerInto(GgufReader.QuantizedTensor quant, float[][] X, float[][] Y, int rows, int cols) {
		for (int b = 0; b < X.length; b++)
			System.arraycopy(LlamaTransformerHandler.matVec(quant, X[b], rows, cols), 0, Y[b], 0, rows);
	}

	private void gqaInto(float[] q, float[] kCache, float[] vCache, int seqLen, float[] out, float[] scores) {
		int H = cfg.numHeads();
		int Hd = cfg.headDim();
		int gqaR = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		java.util.Arrays.fill(out, 0f);

		for (int h = 0; h < H; h++) {
			int kvHead = h / gqaR;
			int qBase = h * Hd;
			int kBase = kvHead * Hd;

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
	}

	private float[][] outputProjectionBatch(float[][] x) {
		int N = x.length;
		int H = cfg.hiddenDim();
		float[][] logits = new float[N][];
		for (int b = 0; b < N; b++)
			logits[b] = outputProjection(x[b]);
		return logits;
	}

	public void setKvAdapter(NodeKVCacheAdapter adapter) {
		this.kvAdapter = adapter;
	}

	@Override
	public void evict(String requestId) {
		kvCacheK.remove(requestId);
		kvCacheV.remove(requestId);
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null)
			a.evict(requestId);
	}

	int kvCacheAllocatedSlots(String requestId) {
		DenseKvTensor[] k = kvCacheK.get(requestId);
		return (k == null || k.length == 0) ? 0 : k[0].capacityTokens();
	}

	private float[] getInitialActivation(ForwardRequest request) {
		if (hasEmbeddings) {
			int[] tokenIds = request.tokenIds();
			int tokenId = tokenIds[tokenIds.length - 1];
			int actualVocab = tokenEmbd.length / cfg.hiddenDim();
			tokenId = Math.max(0, Math.min(tokenId, actualVocab - 1));
			float[] x = new float[cfg.hiddenDim()];
			System.arraycopy(tokenEmbd, tokenId * cfg.hiddenDim(), x, 0, cfg.hiddenDim());
			return x;
		}
		float[] x = new float[request.activations().length];
		System.arraycopy(request.activations(), 0, x, 0, x.length);
		return x;
	}

	private float[] runLayers(float[] x, String requestId, int pos) {
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();

		kvCacheK.putIfAbsent(requestId, newKLayers(L));
		kvCacheV.computeIfAbsent(requestId, k -> newVLayers(L));
		DenseKvTensor[] kCache = kvCacheK.get(requestId);
		DenseKvTensor[] vCache = kvCacheV.get(requestId);

		for (int li = 0; li < L; li++) {
			kCache[li].ensureCapacity(pos);
			vCache[li].ensureCapacity(pos);
		}

		float[] kScratch = null;
		float[] vScratch = null;
		if (cacheTypes.usesQuantized()) {
			kScratch = new float[(pos + 1) * cfg.kvDim()];
			vScratch = new float[(pos + 1) * cfg.kvDim()];
		}

		for (int li = 0; li < L; li++)
			x = transformerLayer(x, li, pos, kCache[li], vCache[li], kScratch, vScratch);

		NodeKVCacheAdapter a = kvAdapter;
		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos,
			DenseKvTensor kCacheLayer, DenseKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());
		float[] attnProj = Qwen3TransformerHandler.attentionLayer(new MoeLayerWeights(li), cfg, xNorm, pos,
				kCacheLayer, vCacheLayer, kScratch, vScratch);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = moeFfn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	private float[] moeFfn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int nExp = cfg.expertCount();
		int topK = cfg.expertUsedCount();
		int expFf = cfg.expertFeedForwardLength();

		float[] router = LlamaTransformerHandler.matVec(ffnGateInp[li], x, nExp, H);
		LlamaTransformerHandler.softmax(router, nExp);

		int[] topExperts = topKIndices(router, topK);
		float[] weights = new float[topK];
		for (int j = 0; j < topK; j++)
			weights[j] = router[topExperts[j]];

		if (cfg.expertWeightsNorm()) {
			float sum = 0f;
			for (float w : weights)
				sum += w;
			if (sum > 0f)
				for (int j = 0; j < topK; j++)
					weights[j] /= sum;
		}

		float[] out = new float[H];
		float scale = cfg.expertWeightsScale();
		for (int j = 0; j < topK; j++) {
			int e = topExperts[j];
			float w = weights[j] * scale;
			float[] gate = matVecExpert(ffnGateExps[li], e, x, expFf, H, nExp);
			float[] up = matVecExpert(ffnUpExps[li], e, x, expFf, H, nExp);
			float[] hidden = new float[expFf];
			for (int i = 0; i < expFf; i++)
				hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
			float[] down = matVecExpert(ffnDownExps[li], e, hidden, H, expFf, nExp);
			for (int i = 0; i < H; i++)
				out[i] += w * down[i];
		}
		return out;
	}

	/**
	 * Matrix-vector for one expert slice from a 3D GGUF tensor stored as
	 * {@code [numExperts, rows, cols]} row-major.
	 */
	static float[] matVecExpert(GgufReader.QuantizedTensor tensor, int expertIdx, float[] x, int rows, int cols,
			int numExperts) {
		int rowStart = expertIdx * rows;
		int rowEnd = rowStart + rows;
		return LlamaTransformerHandler.matVec(tensor, x, rowStart, rowEnd, cols);
	}

	private static int[] topKIndices(float[] scores, int k) {
		int n = scores.length;
		int[] idx = new int[n];
		for (int i = 0; i < n; i++)
			idx[i] = i;
		for (int i = 0; i < Math.min(k, n); i++) {
			int best = i;
			for (int j = i + 1; j < n; j++) {
				if (scores[idx[j]] > scores[idx[best]])
					best = j;
			}
			int tmp = idx[i];
			idx[i] = idx[best];
			idx[best] = tmp;
		}
		return Arrays.copyOf(idx, Math.min(k, n));
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xNorm, actualVocab, cfg.hiddenDim());
	}

	private DenseKvTensor[] newKLayers(int L) {
		return DenseKvTensor.layers(L, cacheTypes.typeK(), cfg.kvDim());
	}

	private DenseKvTensor[] newVLayers(int L) {
		return DenseKvTensor.layers(L, cacheTypes.typeV(), cfg.kvDim());
	}

	private final class MoeLayerWeights implements Qwen3TransformerHandler.Qwen3AttentionWeights {
		private final int li;

		MoeLayerWeights(int li) {
			this.li = li;
		}

		@Override
		public float[] qNorm() {
			return qNorm[li];
		}

		@Override
		public float[] kNorm() {
			return kNorm[li];
		}

		@Override
		public float[] matVecQ(float[] x, int rows, int cols) {
			return LlamaTransformerHandler.matVec(attnQ[li], x, cfg.qDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecK(float[] x, int rows, int cols) {
			return LlamaTransformerHandler.matVec(attnK[li], x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecV(float[] x, int rows, int cols) {
			return LlamaTransformerHandler.matVec(attnV[li], x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecWo(float[] x, int rows, int cols) {
			return LlamaTransformerHandler.matVec(wo[li], x, cfg.hiddenDim(), cfg.qDim());
		}
	}
}