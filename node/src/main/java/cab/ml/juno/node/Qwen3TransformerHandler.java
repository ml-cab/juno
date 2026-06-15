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
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Qwen3 dense transformer forward pass — separate Q/K/V projections with per-head
 * Q/K RMS norms and unfused SwiGLU FFN.
 */
public final class Qwen3TransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Qwen3TransformerHandler.class.getName());
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
	private final GgufReader.QuantizedTensor[] ffnGate;
	private final GgufReader.QuantizedTensor[] ffnUp;
	private final GgufReader.QuantizedTensor[] wDown;

	private final MatVec backend;
	private DeviceHalfMatrix[] attnQDev = null;
	private DeviceHalfMatrix[] attnKDev = null;
	private DeviceHalfMatrix[] attnVDev = null;
	private DeviceHalfMatrix[] woDev = null;
	private DeviceHalfMatrix[] ffnGateDev = null;
	private DeviceHalfMatrix[] ffnUpDev = null;
	private DeviceHalfMatrix[] wDownDev = null;
	private DeviceHalfMatrix outputProjDev = null;

	private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
	private volatile NodeKVCacheAdapter kvAdapter;

	public static Qwen3TransformerHandler load(Path modelPath, ShardContext context) throws IOException {
		return load(modelPath, context, CpuMatVec.INSTANCE);
	}

	public static Qwen3TransformerHandler load(Path modelPath, ShardContext context, MatVec backend)
			throws IOException {
		log.info("Loading Qwen3 GGUF shard: layers " + context.startLayer() + "–" + context.endLayer() + "  backend="
				+ backend.getClass().getSimpleName() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			Qwen3Config config = Qwen3Config.from(r);
			log.info("Model: " + config);
			return new Qwen3TransformerHandler(r, config, context, backend);
		}
	}

	private Qwen3TransformerHandler(GgufReader r, Qwen3Config cfg, ShardContext ctx, MatVec backend)
			throws IOException {
		this.cfg = cfg;
		this.backend = backend;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

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

		int headDim = cfg.headDim();
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
			ffnGate[li] = r.tensorRaw("blk." + i + ".ffn_gate.weight");
			ffnUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");

			if (qNorm[li].length != headDim || kNorm[li].length != headDim) {
				throw new IOException("Layer " + i + ": q/k norm size mismatch (expected headDim=" + headDim + ")");
			}
		}

		if (backend instanceof GpuMatVec cuda) {
			uploadGpuWeights(cuda, L, H, cfg.qDim(), kvDim, I);
		}

		log.info("Qwen3 shard loaded — " + L + " layers");
	}

	private void uploadGpuWeights(GpuMatVec cuda, int L, int H, int qDim, int kvDim, int I) {
		log.info("Uploading Qwen3 dequantized projection weights to GPU (FP16)…");
		DeviceHalfMatrix[] qD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] kD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] vD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] woD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] gD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] uD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] dD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix outD = null;
		try {
			for (int li = 0; li < L; li++) {
				qD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(attnQ[li], qDim, H), qDim, H);
				kD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(attnK[li], kvDim, H), kvDim, H);
				vD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(attnV[li], kvDim, H), kvDim, H);
				woD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(wo[li], H, qDim), H, qDim);
				gD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(ffnGate[li], I, H), I, H);
				uD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(ffnUp[li], I, H), I, H);
				dD[li] = cuda.uploadHalf(LlamaTransformerHandler.dequantize(wDown[li], H, I), H, I);
			}
			if (hasOutputProj) {
				int actualVocab = outputProj.length / H;
				outD = cuda.uploadHalf(outputProj, actualVocab, H);
			}
			this.attnQDev = qD;
			this.attnKDev = kD;
			this.attnVDev = vD;
			this.woDev = woD;
			this.ffnGateDev = gD;
			this.ffnUpDev = uD;
			this.wDownDev = dD;
			this.outputProjDev = outD;
		} catch (IllegalStateException ex) {
			closeDeviceHalfMatrixArray(qD);
			closeDeviceHalfMatrixArray(kD);
			closeDeviceHalfMatrixArray(vD);
			closeDeviceHalfMatrixArray(woD);
			closeDeviceHalfMatrixArray(gD);
			closeDeviceHalfMatrixArray(uD);
			closeDeviceHalfMatrixArray(dD);
			if (outD != null)
				outD.close();
			log.warning("Qwen3 GPU upload failed — using CPU quantised matmul: " + ex.getMessage());
		}
	}

	private static void closeDeviceHalfMatrixArray(DeviceHalfMatrix[] a) {
		if (a == null)
			return;
		for (DeviceHalfMatrix m : a) {
			if (m != null && !m.isClosed())
				m.close();
		}
	}

	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight"))
			return r.tensor("output.weight");
		log.info("output.weight not found — using tied embeddings");
		return r.tensor("token_embd.weight");
	}

	@Override
	public void releaseGpuResources() {
		closeDeviceHalfMatrixArray(attnQDev);
		closeDeviceHalfMatrixArray(attnKDev);
		closeDeviceHalfMatrixArray(attnVDev);
		closeDeviceHalfMatrixArray(woDev);
		closeDeviceHalfMatrixArray(ffnGateDev);
		closeDeviceHalfMatrixArray(ffnUpDev);
		closeDeviceHalfMatrixArray(wDownDev);
		attnQDev = attnKDev = attnVDev = null;
		woDev = ffnGateDev = ffnUpDev = wDownDev = null;
		if (outputProjDev != null && !outputProjDev.isClosed())
			outputProjDev.close();
		outputProjDev = null;
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

		evt.handlerType = "qwen3";
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

	int kvCacheAllocatedSlots(String requestId) {
		float[][] k = kvCacheK.get(requestId);
		return (k == null || k.length == 0) ? 0 : k[0].length / cfg.kvDim();
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

		boolean isNew = kvCacheK.putIfAbsent(requestId, new float[L][INITIAL_SEQ_CAPACITY * kvDim]) == null;
		kvCacheV.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
		float[][] kCache = kvCacheK.get(requestId);
		float[][] vCache = kvCacheV.get(requestId);

		NodeKVCacheAdapter a = kvAdapter;
		if (isNew && pos > 0 && a != null) {
			for (int li = 0; li < L; li++) {
				int absLayer = startLayer + li;
				final int i = li;
				a.tryRestore(requestId, absLayer, kvDim).ifPresent(pair -> {
					ensureKvCapacity(kCache, pair.k().length / kvDim - 1, kvDim);
					ensureKvCapacity(vCache, pair.v().length / kvDim - 1, kvDim);
					System.arraycopy(pair.k(), 0, kCache[i], 0, pair.k().length);
					System.arraycopy(pair.v(), 0, vCache[i], 0, pair.v().length);
				});
			}
		}

		ensureKvCapacity(kCache, pos, kvDim);
		ensureKvCapacity(vCache, pos, kvDim);

		for (int li = 0; li < L; li++)
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);

		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen, kvDim);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());
		float[] attnProj = attentionLayer(xNorm, li, pos, kCacheLayer, vCacheLayer);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = denseFfn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/**
	 * Qwen3 attention with per-head Q/K RMS norms — shared by dense and MoE handlers.
	 */
	static float[] attentionLayer(Qwen3AttentionWeights w, Qwen3Config cfg, float[] xNorm, int pos,
			float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();

		float[] q = w.matVecQ(xNorm, qDim, H);
		float[] k = w.matVecK(xNorm, kvDim, H);
		float[] v = w.matVecV(xNorm, kvDim, H);

		rmsNormPerHead(q, w.qNorm(), cfg.numHeads(), cfg.headDim(), cfg.rmsNormEps());
		rmsNormPerHead(k, w.kNorm(), cfg.numKvHeads(), cfg.headDim(), cfg.rmsNormEps());

		Qwen3Rope.apply(q, pos, cfg.numHeads(), cfg.headDim(), cfg.rope());
		Qwen3Rope.apply(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.rope());

		System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
		System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

		float[] attnOut = gqa(cfg, q, kCacheLayer, vCacheLayer, pos + 1);
		return w.matVecWo(attnOut, H, qDim);
	}

	private float[] attentionLayer(float[] xNorm, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		return attentionLayer(new LayerWeights(li), cfg, xNorm, pos, kCacheLayer, vCacheLayer);
	}

	private float[] denseFfn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = matVecLayer(ffnGate[li], ffnGateDev != null ? ffnGateDev[li] : null, x, I, H);
		float[] up = matVecLayer(ffnUp[li], ffnUpDev != null ? ffnUpDev[li] : null, x, I, H);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		return matVecLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, hidden, H, I);
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		if (outputProjDev != null)
			return backend.sgemv(outputProjDev, xNorm);
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xNorm, actualVocab, cfg.hiddenDim());
	}

	private float[] matVecLayer(GgufReader.QuantizedTensor quant, DeviceHalfMatrix dev, float[] x, int rows,
			int cols) {
		if (dev != null)
			return backend.sgemv(dev, x);
		return LlamaTransformerHandler.matVec(quant, x, rows, cols);
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

	/** Per-head RMS norm: same norm weights applied to each head slice. */
	static void rmsNormPerHead(float[] vec, float[] normW, int nHeads, int headDim, float eps) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			float ss = 0f;
			for (int d = 0; d < headDim; d++) {
				float v = vec[base + d];
				ss += v * v;
			}
			float scale = 1f / (float) Math.sqrt(ss / headDim + eps);
			for (int d = 0; d < headDim; d++)
				vec[base + d] = normW[d] * vec[base + d] * scale;
		}
	}

	static float[] gqa(Qwen3Config cfg, float[] q, float[] kCache, float[] vCache, int seqLen) {
		int H = cfg.numHeads();
		int Hd = cfg.headDim();
		int gqa = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] out = new float[H * Hd];
		float[] scores = new float[seqLen];

		for (int h = 0; h < H; h++) {
			int kvHead = h / gqa;
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
		return out;
	}

	/** Indirection for attention matmul — lets MoE handler reuse {@link #attentionLayer}. */
	interface Qwen3AttentionWeights {
		float[] qNorm();

		float[] kNorm();

		float[] matVecQ(float[] x, int rows, int cols);

		float[] matVecK(float[] x, int rows, int cols);

		float[] matVecV(float[] x, int rows, int cols);

		float[] matVecWo(float[] x, int rows, int cols);
	}

	private final class LayerWeights implements Qwen3AttentionWeights {
		private final int li;

		LayerWeights(int li) {
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
			return matVecLayer(attnQ[li], attnQDev != null ? attnQDev[li] : null, x, cfg.qDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecK(float[] x, int rows, int cols) {
			return matVecLayer(attnK[li], attnKDev != null ? attnKDev[li] : null, x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecV(float[] x, int rows, int cols) {
			return matVecLayer(attnV[li], attnVDev != null ? attnVDev[li] : null, x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecWo(float[] x, int rows, int cols) {
			return matVecLayer(wo[li], woDev != null ? woDev[li] : null, x, cfg.hiddenDim(), cfg.qDim());
		}
	}
}
