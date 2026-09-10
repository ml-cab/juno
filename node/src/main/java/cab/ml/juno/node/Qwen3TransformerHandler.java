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

import cab.ml.juno.kvcache.SessionKvLayout;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Qwen3 dense transformer forward pass — separate Q/K/V projections with per-head
 * Q/K RMS norms and unfused SwiGLU FFN.
 */
public final class Qwen3TransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Qwen3TransformerHandler.class.getName());
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

	/** Packed Q4_K residency when {@code --mmq} prefers fused GEMV. */
	private DeviceQ4KMatrix[] attnQQ4Dev = null;
	private DeviceQ4KMatrix[] attnKQ4Dev = null;
	private DeviceQ4KMatrix[] attnVQ4Dev = null;
	private DeviceQ4KMatrix[] woQ4Dev = null;
	private DeviceQ4KMatrix[] ffnGateQ4Dev = null;
	private DeviceQ4KMatrix[] ffnUpQ4Dev = null;
	private DeviceQ4KMatrix[] wDownQ4Dev = null;

	private final int gpuLayersResolved;

	private final Map<String, SessionKvTensor[]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, SessionKvTensor[]> kvCacheV = new ConcurrentHashMap<>();
	private final SessionKvLayout kvLayout;
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
		this.kvLayout = SessionKvLayout.fromEnv(cfg.kvDim());
		log.info(kvLayout.policySummary());

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

		int resolved = 0;
		if (backend instanceof GpuMatVec cuda) {
			resolved = uploadGpuWeights(cuda, L, H, cfg.qDim(), kvDim, I);
		}
		this.gpuLayersResolved = resolved;

		log.info("Qwen3 shard loaded — " + L + " layers");
	}

	private int uploadGpuWeights(GpuMatVec cuda, int L, int H, int qDim, int kvDim, int I) {
		GpuLayerOffload policy = GpuLayerOffload.fromEnv();
		int totalLayers = cfg.numLayers();
		boolean tryMmq = MmqOptions.fromEnv().preferMmq() && cuda.supportsQ4KMmq();
		if (tryMmq)
			log.info("Fused Q4_K MMQ enabled (mmq=" + MmqOptions.fromEnv().policyLabel() + ")");
		log.info("Uploading Qwen3 projection weights to GPU (FP16"
				+ (tryMmq ? ", Q4_K packed when available" : "")
				+ ", gpu-layers=" + policy.policyLabel(totalLayers) + ")…");
		DeviceHalfMatrix[] qD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] kD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] vD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] woD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] gD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] uD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] dD = new DeviceHalfMatrix[L];
		DeviceQ4KMatrix[] qQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] kQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] vQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] woQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] gQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] uQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] dQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceHalfMatrix outD = null;
		int resolvedGlobal = 0;
		try {
			for (int li = 0; li < L; li++) {
				int global = startLayer + li;
				if (!policy.isAuto() && !policy.residentForGlobalLayer(global, totalLayers))
					continue;
				try {
					Q4KResidentUpload.uploadInto(cuda, attnQ[li], qDim, H, tryMmq, li, qD, qQ4);
					Q4KResidentUpload.uploadInto(cuda, attnK[li], kvDim, H, tryMmq, li, kD, kQ4);
					Q4KResidentUpload.uploadInto(cuda, attnV[li], kvDim, H, tryMmq, li, vD, vQ4);
					Q4KResidentUpload.uploadInto(cuda, wo[li], H, qDim, tryMmq, li, woD, woQ4);
					Q4KResidentUpload.uploadInto(cuda, ffnGate[li], I, H, tryMmq, li, gD, gQ4);
					Q4KResidentUpload.uploadInto(cuda, ffnUp[li], I, H, tryMmq, li, uD, uQ4);
					Q4KResidentUpload.uploadInto(cuda, wDown[li], H, I, tryMmq, li, dD, dQ4);
					resolvedGlobal = Math.max(resolvedGlobal, global + 1);
				} catch (IllegalStateException ex) {
					if (!qwen3HandleLayerOom(policy, ex, global))
						throw ex;
					break;
				}
			}
			boolean allLayersGpu = policy.isAuto()
					? resolvedGlobal >= totalLayers
					: policy.residentOutputProjection(totalLayers);
			if (hasOutputProj && allLayersGpu) {
				try {
					int actualVocab = outputProj.length / H;
					outD = cuda.uploadHalf(outputProj, actualVocab, H);
				} catch (IllegalStateException ex) {
					if (!GpuLayerOffload.isVramOom(ex))
						throw ex;
					log.warning("Qwen3: OOM uploading output projection — using CPU matmul");
				}
			}
			this.attnQDev = qD;
			this.attnKDev = kD;
			this.attnVDev = vD;
			this.woDev = woD;
			this.ffnGateDev = gD;
			this.ffnUpDev = uD;
			this.wDownDev = dD;
			this.attnQQ4Dev = qQ4;
			this.attnKQ4Dev = kQ4;
			this.attnVQ4Dev = vQ4;
			this.woQ4Dev = woQ4;
			this.ffnGateQ4Dev = gQ4;
			this.ffnUpQ4Dev = uQ4;
			this.wDownQ4Dev = dQ4;
			this.outputProjDev = outD;
			if (policy.isAuto())
				policy = policy.withAutoResolved(resolvedGlobal);
			log.info("Qwen3 GPU weight upload complete (resolved gpu-layers="
					+ policy.resolvedCount(totalLayers)
					+ (tryMmq ? "+Q4K" : "") + ").");
			return policy.resolvedCount(totalLayers);
		} catch (IllegalStateException ex) {
			closeDeviceHalfMatrixArray(qD);
			closeDeviceHalfMatrixArray(kD);
			closeDeviceHalfMatrixArray(vD);
			closeDeviceHalfMatrixArray(woD);
			closeDeviceHalfMatrixArray(gD);
			closeDeviceHalfMatrixArray(uD);
			closeDeviceHalfMatrixArray(dD);
			Q4KResidentUpload.closeArray(qQ4);
			Q4KResidentUpload.closeArray(kQ4);
			Q4KResidentUpload.closeArray(vQ4);
			Q4KResidentUpload.closeArray(woQ4);
			Q4KResidentUpload.closeArray(gQ4);
			Q4KResidentUpload.closeArray(uQ4);
			Q4KResidentUpload.closeArray(dQ4);
			if (outD != null)
				outD.close();
			if (policy.mode() == GpuLayerOffload.Mode.ALL && GpuLayerOffload.isVramOom(ex)) {
				log.warning("Qwen3 GPU upload failed — using CPU quantised matmul: " + ex.getMessage());
				return 0;
			}
			throw ex;
		}
	}

	private boolean qwen3HandleLayerOom(GpuLayerOffload policy, IllegalStateException ex, int globalLayer) {
		if (!GpuLayerOffload.isVramOom(ex))
			return false;
		if (policy.mode() == GpuLayerOffload.Mode.ALL)
			return false;
		log.warning("Qwen3: OOM at global layer " + globalLayer + " — partial GPU offload");
		return true;
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
		Q4KResidentUpload.closeArray(attnQQ4Dev);
		Q4KResidentUpload.closeArray(attnKQ4Dev);
		Q4KResidentUpload.closeArray(attnVQ4Dev);
		Q4KResidentUpload.closeArray(woQ4Dev);
		Q4KResidentUpload.closeArray(ffnGateQ4Dev);
		Q4KResidentUpload.closeArray(ffnUpQ4Dev);
		Q4KResidentUpload.closeArray(wDownQ4Dev);
		attnQQ4Dev = attnKQ4Dev = attnVQ4Dev = null;
		woQ4Dev = ffnGateQ4Dev = ffnUpQ4Dev = wDownQ4Dev = null;
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
		evt.gpuLayers = gpuLayersResolved;
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
	public BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
		long start = System.nanoTime();
		int W = request.windowSize();
		int H = cfg.hiddenDim();

		float[][] x;
		if (hasEmbeddings && request.isFirstNode()) {
			x = new float[W][H];
			int actualVocab = tokenEmbd.length / H;
			for (int b = 0; b < W; b++) {
				int tokenId = Math.max(0, Math.min(request.tokenIds()[b], actualVocab - 1));
				System.arraycopy(tokenEmbd, tokenId * H, x[b], 0, H);
			}
		} else {
			x = new float[W][H];
			float[] flat = request.activations();
			for (int b = 0; b < W; b++)
				System.arraycopy(flat, b * H, x[b], 0, H);
		}

		x = runLayersBatch(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			float[] logits = outputProjection(x[W - 1]);
			return new BatchForwardResult(request.requestId(), null, logits, W, System.nanoTime() - start);
		}

		float[] flat = new float[W * H];
		for (int b = 0; b < W; b++)
			System.arraycopy(x[b], 0, flat, b * H, H);
		return new BatchForwardResult(request.requestId(), flat, null, W, System.nanoTime() - start);
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

	private float[][] runLayersBatch(float[][] x, String requestId, int startPos) {
		int W = x.length;
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();
		int lastPos = startPos + W - 1;

		boolean isNew = kvCacheK.putIfAbsent(requestId, newKLayers(L)) == null;
		kvCacheV.computeIfAbsent(requestId, k -> newVLayers(L));
		SessionKvTensor[] kCache = kvCacheK.get(requestId);
		SessionKvTensor[] vCache = kvCacheV.get(requestId);

		NodeKVCacheAdapter a = kvAdapter;
		if (isNew && startPos > 0 && a != null) {
			for (int li = 0; li < L; li++) {
				int absLayer = startLayer + li;
				final int i = li;
				a.tryRestore(requestId, absLayer, kvDim).ifPresent(pair ->
						restoreLayer(kCache[i], vCache[i], pair, kvDim));
			}
		}

		for (int li = 0; li < L; li++) {
			kCache[li].ensureCapacity(lastPos);
			vCache[li].ensureCapacity(lastPos);
		}

		BatchWorkspace ws = new BatchWorkspace(W, cfg.hiddenDim(), cfg.qDim(), cfg.intermediateSize(),
				cfg.kvDim(), cfg.numHeads(), lastPos + 1, kvLayout.needsAttentionScratch());

		for (int li = 0; li < L; li++)
			x = transformerLayerBatch(x, li, startPos, kCache[li], vCache[li], ws);

		if (a != null) {
			int seqLen = lastPos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
		}
		return x;
	}

	private float[][] runLayersMultiDecode(float[][] x, java.util.List<String> requestIds, int[] positions) {
		int N = x.length;
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();
		int maxPos = 0;
		for (int pos : positions)
			maxPos = Math.max(maxPos, pos);

		SessionKvTensor[][] kCaches = new SessionKvTensor[N][];
		SessionKvTensor[][] vCaches = new SessionKvTensor[N][];

		NodeKVCacheAdapter a = kvAdapter;
		for (int i = 0; i < N; i++) {
			String requestId = requestIds.get(i);
			int pos = positions[i];

			boolean isNew = kvCacheK.putIfAbsent(requestId, newKLayers(L)) == null;
			kvCacheV.computeIfAbsent(requestId, k -> newVLayers(L));
			SessionKvTensor[] kCache = kvCacheK.get(requestId);
			SessionKvTensor[] vCache = kvCacheV.get(requestId);

			if (isNew && pos > 0 && a != null) {
				for (int li = 0; li < L; li++) {
					int absLayer = startLayer + li;
					final int layerIdx = li;
					a.tryRestore(requestId, absLayer, kvDim).ifPresent(pair ->
							restoreLayer(kCache[layerIdx], vCache[layerIdx], pair, kvDim));
				}
			}

			for (int li = 0; li < L; li++) {
				kCache[li].ensureCapacity(pos);
				vCache[li].ensureCapacity(pos);
			}
			kCaches[i] = kCache;
			vCaches[i] = vCache;
		}

		BatchWorkspace ws = new BatchWorkspace(N, cfg.hiddenDim(), cfg.qDim(), cfg.intermediateSize(),
				cfg.kvDim(), cfg.numHeads(), maxPos + 1, kvLayout.needsAttentionScratch());

		for (int li = 0; li < L; li++) {
			SessionKvTensor[] kLayers = new SessionKvTensor[N];
			SessionKvTensor[] vLayers = new SessionKvTensor[N];
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

	private static final class BatchWorkspace {
		final float[][] norm1, norm2, q, k, v, attnOut, attnProj, gate, up, hidden, ffnOut;
		final float[] scores;
		final float[] kDequant, vDequant;

		BatchWorkspace(int W, int H, int qDim, int I, int kvDim, int numHeads, int maxSeqLen, boolean quantKv) {
			norm1 = new float[W][H];
			norm2 = new float[W][H];
			q = new float[W][qDim];
			k = new float[W][kvDim];
			v = new float[W][kvDim];
			attnOut = new float[W][qDim];
			attnProj = new float[W][H];
			gate = new float[W][I];
			up = new float[W][I];
			hidden = new float[W][I];
			ffnOut = new float[W][H];
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

	private float[][] transformerLayerBatch(float[][] x, int li, int startPos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer, BatchWorkspace ws) {
		int W = x.length;
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);

		sgemmLayerInto(attnQ[li], attnQQ4Dev, attnQDev, li, ws.norm1, ws.q, qDim, H);
		sgemmLayerInto(attnK[li], attnKQ4Dev, attnKDev, li, ws.norm1, ws.k, kvDim, H);
		sgemmLayerInto(attnV[li], attnVQ4Dev, attnVDev, li, ws.norm1, ws.v, kvDim, H);

		for (int b = 0; b < W; b++) {
			rmsNormPerHead(ws.q[b], qNorm[li], cfg.numHeads(), cfg.headDim(), cfg.rmsNormEps());
			rmsNormPerHead(ws.k[b], kNorm[li], cfg.numKvHeads(), cfg.headDim(), cfg.rmsNormEps());
		}

		for (int b = 0; b < W; b++) {
			Qwen3Rope.apply(ws.q[b], startPos + b, cfg.numHeads(), cfg.headDim(), cfg.rope());
			Qwen3Rope.apply(ws.k[b], startPos + b, cfg.numKvHeads(), cfg.headDim(), cfg.rope());
		}

		for (int b = 0; b < W; b++) {
			kCacheLayer.writeToken(startPos + b, ws.k[b]);
			vCacheLayer.writeToken(startPos + b, ws.v[b]);
		}

		for (int b = 0; b < W; b++) {
			int seqLen = startPos + b + 1;
			float[] kView = kCacheLayer.viewForAttention(seqLen, ws.kDequant);
			float[] vView = vCacheLayer.viewForAttention(seqLen, ws.vDequant);
			gqaInto(cfg, ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
		}

		sgemmLayerInto(wo[li], woQ4Dev, woDev, li, ws.attnOut, ws.attnProj, H, qDim);

		for (int b = 0; b < W; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.attnProj[b][d];

		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);

		sgemmLayerInto(ffnGate[li], ffnGateQ4Dev, ffnGateDev, li, ws.norm2, ws.gate, I, H);
		sgemmLayerInto(ffnUp[li], ffnUpQ4Dev, ffnUpDev, li, ws.norm2, ws.up, I, H);

		for (int b = 0; b < W; b++)
			for (int i = 0; i < I; i++)
				ws.hidden[b][i] = LlamaTransformerHandler.silu(ws.gate[b][i]) * ws.up[b][i];

		sgemmLayerInto(wDown[li], wDownQ4Dev, wDownDev, li, ws.hidden, ws.ffnOut, H, I);

		for (int b = 0; b < W; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.ffnOut[b][d];

		return x;
	}

	private float[][] transformerLayerMultiDecode(float[][] x, int li, int[] positions,
			SessionKvTensor[] kCacheLayers, SessionKvTensor[] vCacheLayers, BatchWorkspace ws) {
		int N = x.length;
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		for (int b = 0; b < N; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);

		sgemmLayerInto(attnQ[li], attnQQ4Dev, attnQDev, li, ws.norm1, ws.q, qDim, H);
		sgemmLayerInto(attnK[li], attnKQ4Dev, attnKDev, li, ws.norm1, ws.k, kvDim, H);
		sgemmLayerInto(attnV[li], attnVQ4Dev, attnVDev, li, ws.norm1, ws.v, kvDim, H);

		for (int b = 0; b < N; b++) {
			rmsNormPerHead(ws.q[b], qNorm[li], cfg.numHeads(), cfg.headDim(), cfg.rmsNormEps());
			rmsNormPerHead(ws.k[b], kNorm[li], cfg.numKvHeads(), cfg.headDim(), cfg.rmsNormEps());
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
			gqaInto(cfg, ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
		}

		sgemmLayerInto(wo[li], woQ4Dev, woDev, li, ws.attnOut, ws.attnProj, H, qDim);

		for (int b = 0; b < N; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.attnProj[b][d];

		for (int b = 0; b < N; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);

		sgemmLayerInto(ffnGate[li], ffnGateQ4Dev, ffnGateDev, li, ws.norm2, ws.gate, I, H);
		sgemmLayerInto(ffnUp[li], ffnUpQ4Dev, ffnUpDev, li, ws.norm2, ws.up, I, H);

		for (int b = 0; b < N; b++)
			for (int i = 0; i < I; i++)
				ws.hidden[b][i] = LlamaTransformerHandler.silu(ws.gate[b][i]) * ws.up[b][i];

		sgemmLayerInto(wDown[li], wDownQ4Dev, wDownDev, li, ws.hidden, ws.ffnOut, H, I);

		for (int b = 0; b < N; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.ffnOut[b][d];

		return x;
	}

	private void sgemmLayerInto(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4,
			DeviceHalfMatrix[] half, int li, float[][] X, float[][] Y, int rows, int cols) {
		if (q4 != null && q4[li] != null) {
			float[][] tmp = backend.sgemm(q4[li], X);
			for (int b = 0; b < X.length; b++)
				System.arraycopy(tmp[b], 0, Y[b], 0, rows);
			return;
		}
		if (half != null && half[li] != null) {
			float[][] tmp = backend.sgemm(half[li], X);
			for (int b = 0; b < X.length; b++)
				System.arraycopy(tmp[b], 0, Y[b], 0, rows);
			return;
		}
		for (int b = 0; b < X.length; b++)
			System.arraycopy(LlamaTransformerHandler.matVec(quant, X[b], rows, cols), 0, Y[b], 0, rows);
	}

	private static void gqaInto(Qwen3Config cfg, float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores) {
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
		float[][] xNorm = new float[N][];
		for (int b = 0; b < N; b++)
			xNorm[b] = LlamaTransformerHandler.rmsNorm(x[b], outputNorm, cfg.rmsNormEps());
		if (outputProjDev != null)
			return backend.sgemm(outputProjDev, xNorm);
		int actualVocab = outputProj.length / H;
		float[][] logits = new float[N][];
		for (int b = 0; b < N; b++)
			logits[b] = LlamaTransformerHandler.matVec(outputProj, xNorm[b], actualVocab, H);
		return logits;
	}

	public void setKvAdapter(NodeKVCacheAdapter adapter) {
		this.kvAdapter = adapter;
		if (adapter != null)
			adapter.manager().pagedArena().ifPresent(kvLayout::bindSharedArena);
	}

	@Override
	public void evict(String requestId) {
		SessionKvLayout.releaseLayers(kvCacheK.remove(requestId));
		SessionKvLayout.releaseLayers(kvCacheV.remove(requestId));
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null)
			a.evict(requestId);
	}

	int kvCacheAllocatedSlots(String requestId) {
		SessionKvTensor[] k = kvCacheK.get(requestId);
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

		boolean isNew = kvCacheK.putIfAbsent(requestId, newKLayers(L)) == null;
		kvCacheV.computeIfAbsent(requestId, k -> newVLayers(L));
		SessionKvTensor[] kCache = kvCacheK.get(requestId);
		SessionKvTensor[] vCache = kvCacheV.get(requestId);

		NodeKVCacheAdapter a = kvAdapter;
		if (isNew && pos > 0 && a != null) {
			for (int li = 0; li < L; li++) {
				int absLayer = startLayer + li;
				final int i = li;
				a.tryRestore(requestId, absLayer, kvDim).ifPresent(pair ->
						restoreLayer(kCache[i], vCache[i], pair, kvDim));
			}
		}

		for (int li = 0; li < L; li++) {
			kCache[li].ensureCapacity(pos);
			vCache[li].ensureCapacity(pos);
		}

		float[] kScratch = null;
		float[] vScratch = null;
		if (kvLayout.needsAttentionScratch()) {
			kScratch = new float[(pos + 1) * kvDim];
			vScratch = new float[(pos + 1) * kvDim];
		}

		for (int li = 0; li < L; li++)
			x = transformerLayer(x, li, pos, kCache[li], vCache[li], kScratch, vScratch);

		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());
		float[] attnProj = attentionLayer(xNorm, li, pos, kCacheLayer, vCacheLayer, kScratch, vScratch);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = denseFfn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/**
	 * Qwen3 attention with per-head Q/K RMS norms — shared by dense and MoE handlers.
	 */
	static float[] attentionLayer(Qwen3AttentionWeights w, Qwen3Config cfg, float[] xNorm, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
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

		kCacheLayer.writeToken(pos, k);
		vCacheLayer.writeToken(pos, v);

		int seqLen = pos + 1;
		float[] kView = kCacheLayer.viewForAttention(seqLen, kScratch);
		float[] vView = vCacheLayer.viewForAttention(seqLen, vScratch);
		float[] attnOut = gqa(cfg, q, kView, vView, seqLen);
		return w.matVecWo(attnOut, H, qDim);
	}

	private float[] attentionLayer(float[] xNorm, int li, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
		return attentionLayer(new LayerWeights(li), cfg, xNorm, pos, kCacheLayer, vCacheLayer, kScratch, vScratch);
	}

	private float[] denseFfn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = matVecLayer(ffnGate[li], ffnGateQ4Dev, ffnGateDev, li, x, I, H);
		float[] up = matVecLayer(ffnUp[li], ffnUpQ4Dev, ffnUpDev, li, x, I, H);
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		return matVecLayer(wDown[li], wDownQ4Dev, wDownDev, li, hidden, H, I);
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		if (outputProjDev != null)
			return backend.sgemv(outputProjDev, xNorm);
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xNorm, actualVocab, cfg.hiddenDim());
	}

	private float[] matVecLayer(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4, DeviceHalfMatrix[] half,
			int li, float[] x, int rows, int cols) {
		if (q4 != null && q4[li] != null)
			return backend.sgemv(q4[li], x);
		if (half != null && half[li] != null)
			return backend.sgemv(half[li], x);
		return LlamaTransformerHandler.matVec(quant, x, rows, cols);
	}

	private SessionKvTensor[] newKLayers(int L) {
		return kvLayout.newKLayers(L);
	}

	private SessionKvTensor[] newVLayers(int L) {
		return kvLayout.newVLayers(L);
	}

	private void restoreLayer(SessionKvTensor k, SessionKvTensor v, NodeKVCacheAdapter.KvPair pair, int kvDim) {
		int seqLen = pair.k().length / kvDim;
		k.loadFloatPrefix(pair.k(), seqLen);
		v.loadFloatPrefix(pair.v(), seqLen);
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
			return matVecLayer(attnQ[li], attnQQ4Dev, attnQDev, li, x, cfg.qDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecK(float[] x, int rows, int cols) {
			return matVecLayer(attnK[li], attnKQ4Dev, attnKDev, li, x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecV(float[] x, int rows, int cols) {
			return matVecLayer(attnV[li], attnVQ4Dev, attnVDev, li, x, cfg.kvDim(), cfg.hiddenDim());
		}

		@Override
		public float[] matVecWo(float[] x, int rows, int cols) {
			return matVecLayer(wo[li], woQ4Dev, woDev, li, x, cfg.hiddenDim(), cfg.qDim());
		}
	}
}