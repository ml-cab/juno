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

/**
 * Qwen3-MoE transformer forward pass — same Q/K-norm attention as
 * {@link Qwen3TransformerHandler} with routed expert SwiGLU FFN.
 */
public final class Qwen3MoeTransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Qwen3MoeTransformerHandler.class.getName());
	private static final int MAX_SEQ_LEN = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64;

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
	private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
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

	public void setKvAdapter(NodeKVCacheAdapter adapter) {
		this.kvAdapter = adapter;
	}

	public void evict(String requestId) {
		kvCacheK.remove(requestId);
		kvCacheV.remove(requestId);
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null)
			a.evict(requestId);
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

		kvCacheK.putIfAbsent(requestId, new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		kvCacheV.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		float[][] kCache = kvCacheK.get(requestId);
		float[][] vCache = kvCacheV.get(requestId);

		ensureKvCapacity(kCache, pos, kvDim);
		ensureKvCapacity(vCache, pos, kvDim);

		for (int li = 0; li < L; li++)
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);

		NodeKVCacheAdapter a = kvAdapter;
		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen, kvDim);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());
		float[] attnProj = Qwen3TransformerHandler.attentionLayer(new MoeLayerWeights(li), cfg, xNorm, pos,
				kCacheLayer, vCacheLayer);
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
