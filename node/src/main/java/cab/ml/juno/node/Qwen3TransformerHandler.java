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
	private final DeviceMatVecFallback matVecFallback = new DeviceMatVecFallback("Qwen3");
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

	/**
	 * GPU-resident attention kernel and its per-request device KV mirrors
	 * ({@code --gpu-attention}); {@code null} on the CPU backend, with the flag off,
	 * or on a backend without the kernel. Cleared by {@link #releaseGpuResources()}.
	 */
	private GpuAttentionMirror gpuAttention;

	/**
	 * The prefill-window device region (norms including the per-head Q/K norm, matmuls,
	 * RoPE where the file's rotation has a device kernel, attention where the GPU
	 * attention mirror is active, SwiGLU and residual adds on the device), or null on the
	 * CPU backend, when turned off, or after {@link #releaseGpuResources}.
	 */
	private PrefillWindowRegion prefillRegion;

	/**
	 * The device-resident decode region ({@code --gpu-residency}): norm, the Q/K/V
	 * projections, the per-head Q/K norms, split-half RoPE, the KV append, attention
	 * and, where the layer's weights are on the device, the rest of the layer, with
	 * the residual row kept on the device between layers. Null when not requested or
	 * not runnable here, and after {@link #releaseGpuResources}.
	 */
	private ResidentQkvPath residentQkv;

	/** Guards {@link #warnPrefillRegionFellBackOnce} so the hot path logs once. */
	private final java.util.concurrent.atomic.AtomicBoolean prefillRegionFallbackWarned =
			new java.util.concurrent.atomic.AtomicBoolean();

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

	/**
	 * Test-only: load with a forced RoPE pair layout, so a test can compare a
	 * file under both layouts. There is deliberately no flag or property that
	 * reaches this.
	 */
	static Qwen3TransformerHandler load(Path modelPath, ShardContext context, MatVec backend,
			RopePairing ropePairing) throws IOException {
		try (GgufReader r = GgufReader.open(modelPath)) {
			Qwen3Config config = Qwen3Config.from(r).withRopePairing(ropePairing);
			log.info("Model: " + config + "  ropePairing=" + ropePairing + " (forced)");
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
		this.gpuAttention = GpuAttentionMirror.open(backend, "Qwen3", L, kvDim, cfg.numHeads(), cfg.headDim(),
				cfg.gqaRatio());
		this.prefillRegion = openPrefillRegion(backend, L);
		this.residentQkv = GpuResidencyOptions.fromEnv().requested() ? openResidentQkv(backend, L) : null;

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

	/**
	 * Builds the prefill-window device region over every layer whose seven projections
	 * are on the device, with the per-head Q/K norm. RoPE runs on the device unless the
	 * file declares YaRN scaling (no device kernel; the host rotates), and attention moves
	 * into the region whenever RoPE does and the GPU attention mirror is active.
	 */
	private PrefillWindowRegion openPrefillRegion(MatVec backend, int L) {
		if (!(backend instanceof CudaMatVec))
			return null;
		PrefillWindowRegion.Layer[] layers = new PrefillWindowRegion.Layer[L];
		for (int li = 0; li < L; li++)
			layers[li] = PrefillWindowRegion.Layer.separate(PrefillWindowRegion.Matrix.of(attnQQ4Dev, attnQDev, li),
					PrefillWindowRegion.Matrix.of(attnKQ4Dev, attnKDev, li),
					PrefillWindowRegion.Matrix.of(attnVQ4Dev, attnVDev, li), PrefillWindowRegion.Matrix.of(woQ4Dev, woDev, li),
					PrefillWindowRegion.Matrix.of(ffnGateQ4Dev, ffnGateDev, li),
					PrefillWindowRegion.Matrix.of(ffnUpQ4Dev, ffnUpDev, li),
					PrefillWindowRegion.Matrix.of(wDownQ4Dev, wDownDev, li), attnNorm[li], ffnNorm[li], null, null, null);
		for (int li = 0; li < L; li++)
			layers[li] = PrefillWindowRegion.Layer.withHeadNorms(layers[li], qNorm[li], kNorm[li]);
		PrefillWindowRegion.Shape shape = new PrefillWindowRegion.Shape(cfg.hiddenDim(), cfg.qDim(), cfg.kvDim(),
				cfg.intermediateSize(), cfg.numHeads(), cfg.numKvHeads(), cfg.headDim(), cfg.gqaRatio(),
				cfg.rmsNormEps());
		Qwen3RopeConfig rope = cfg.rope();
		return PrefillWindowRegion.create("Qwen3", backend, shape, layers, rope.yarn() ? null : rope.pairing(),
				rope.freqBase(), gpuAttention != null);
	}

	@Override
	public boolean prefillRegionActive() {
		return prefillRegion != null;
	}

	@Override
	public long prefillWindowDeviceBytes(int rows) {
		PrefillWindowRegion region = prefillRegion;
		return region == null ? 0L : region.windowDeviceBytes(rows);
	}

	@Override
	public void releaseGpuResources() {
		// The regions hold references to the device matrices below; close them first.
		if (residentQkv != null)
			residentQkv.close();
		residentQkv = null;
		if (prefillRegion != null)
			prefillRegion.close();
		prefillRegion = null;
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
		GpuAttentionMirror g = gpuAttention;
		gpuAttention = null;
		if (g != null)
			g.close();
	}

	@Override
	public boolean gpuAttentionActive() {
		return gpuAttention != null;
	}

	/** Whether the device-resident decode region ({@code --gpu-residency}) is active for this handler. */
	boolean gpuResidencyActive() {
		return residentQkv != null;
	}

	/** Whether the decode region runs whole layers: it attends on the device and at least one layer runs whole. */
	boolean gpuResidencyWholeLayerActive() {
		ResidentQkvPath path = residentQkv;
		if (path == null || gpuAttention == null || !path.attendsOnDevice())
			return false;
		for (int li = 0; li < endLayer - startLayer; li++)
			if (path.runsWholeLayer(li))
				return true;
		return false;
	}

	/**
	 * Builds the device-resident decode region when this model and backend can run
	 * it, and otherwise says once why not and returns null (today's path). The
	 * per-head Q/K norms run in the region; YaRN-scaled RoPE does not, and declines it.
	 */
	private ResidentQkvPath openResidentQkv(MatVec backend, int L) {
		if (!(backend instanceof GpuMatVec cuda)) {
			GpuResidencyOptions.announceIfExplicit(log, "the CPU backend", "has no device to keep activations on");
			return null;
		}
		String reason = ResidentQkvPath.unsupportedReason(cuda.gpuContext(), cfg.base());
		if (reason == null && cfg.rope().yarn())
			reason = "uses YaRN-scaled RoPE, which the device RoPE kernel does not implement";
		if (reason != null) {
			GpuResidencyOptions.announceUnsupported(log, "architecture " + cfg.base().architecture(), reason);
			return null;
		}
		if (attnQQ4Dev == null || attnKQ4Dev == null || attnVQ4Dev == null) {
			GpuResidencyOptions.announceUnsupported(log, "this model's Q/K/V projections",
					"are not K-quant MMQ matrices on the device (needs a K-quant file and --mmq on or auto)");
			return null;
		}
		CudaRope rope = CudaRope.tryCreate(cuda.gpuContext(), cfg.headDim(), cfg.rope().freqBase(),
				cfg.rope().pairing());
		ResidentQkvPath path;
		try {
			path = ResidentQkvPath.create(cuda.gpuContext(), cfg.base(), attnNorm, attnQQ4Dev, attnKQ4Dev, attnVQ4Dev,
					new ResidentLayerTail.Weights(ffnNorm, woQ4Dev, ffnGateQ4Dev, ffnUpQ4Dev, wDownQ4Dev), rope,
					new ResidentQkvPath.HeadNorms(qNorm, kNorm));
		} catch (RuntimeException e) {
			GpuResidencyOptions.announceUnsupported(log, "the device-resident decode region",
					"could not be built (" + e.getMessage() + ")");
			return null;
		}
		return ResidentQkvPath.activate(log, path, L, gpuAttention != null);
	}

	/** Whether layer {@code li}'s Q/K/V projections run on the device (a layer the KV mirror serves). */
	private boolean layerOnDevice(int li) {
		return (attnQQ4Dev != null && attnQQ4Dev[li] != null) || (attnQDev != null && attnQDev[li] != null);
	}

	/** The request's device KV mirrors, or {@code null} when the kernel path is not active. */
	private DeviceKvCache[] mirrorsFor(String requestId) {
		GpuAttentionMirror g = gpuAttention;
		return g != null ? g.layersFor(requestId) : null;
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

		WindowStepEvent embedEvt = WindowStepEvent.start();
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
		embedEvt.end(WindowStepEvent.EMBED, W, request.startPosition());

		x = runLayersBatch(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			WindowStepEvent headEvt = WindowStepEvent.start();
			float[] logits = outputProjection(x[W - 1]);
			headEvt.end(WindowStepEvent.LM_HEAD, W, request.startPosition());
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
		DeviceKvCache[] mirrors = mirrorsFor(requestId);

		PrefillWindowRegion.Window win = openPrefillWindow(W);
		try {
			for (int li = 0; li < L; li++)
				x = transformerLayerBatch(x, li, startPos, kCache[li], vCache[li], ws,
						GpuAttentionMirror.layer(mirrors, li, layerOnDevice(li)), win);
			if (win != null)
				materializeResidual(win, x, startPos);
		} finally {
			if (win != null)
				win.close();
		}

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
		DeviceKvCache[][] mirrors = new DeviceKvCache[N][];

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
			mirrors[i] = mirrorsFor(requestId);
		}

		BatchWorkspace ws = new BatchWorkspace(N, cfg.hiddenDim(), cfg.qDim(), cfg.intermediateSize(),
				cfg.kvDim(), cfg.numHeads(), maxPos + 1, kvLayout.needsAttentionScratch());

		for (int li = 0; li < L; li++) {
			SessionKvTensor[] kLayers = new SessionKvTensor[N];
			SessionKvTensor[] vLayers = new SessionKvTensor[N];
			DeviceKvCache[] devLayers = new DeviceKvCache[N];
			boolean onDevice = layerOnDevice(li);
			for (int i = 0; i < N; i++) {
				kLayers[i] = kCaches[i][li];
				vLayers[i] = vCaches[i][li];
				devLayers[i] = GpuAttentionMirror.layer(mirrors[i], li, onDevice);
			}
			x = transformerLayerMultiDecode(x, li, positions, kLayers, vLayers, ws, devLayers);
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
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer, BatchWorkspace ws, DeviceKvCache mirror,
			PrefillWindowRegion.Window win) {
		if (win != null && prefillRegion != null && prefillRegion.eligible(li))
			return transformerLayerOnDevice(x, li, startPos, kCacheLayer, vCacheLayer, ws, mirror, win);
		if (win != null)
			materializeResidual(win, x, startPos); // this layer reads the residual on the host
		int W = x.length;
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		RmsNormEvent normEvt1 = new RmsNormEvent();
		normEvt1.begin();
		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);
		normEvt1.windowSize = W;
		normEvt1.startPosition = startPos;
		normEvt1.dimension = H;
		normEvt1.commit();

		projectWindow(attnQ[li], attnQQ4Dev, attnQDev, li, ws.norm1, ws.q, qDim, H, startPos);
		projectWindow(attnK[li], attnKQ4Dev, attnKDev, li, ws.norm1, ws.k, kvDim, H, startPos);
		projectWindow(attnV[li], attnVQ4Dev, attnVDev, li, ws.norm1, ws.v, kvDim, H, startPos);

		normalizeAndRotateQk(li, startPos, ws, W);
		writeKvAndAttend(startPos, kCacheLayer, vCacheLayer, ws, mirror, W);

		projectWindow(wo[li], woQ4Dev, woDev, li, ws.attnOut, ws.attnProj, H, qDim, startPos);

		ResidualAddEvent residEvt1 = new ResidualAddEvent();
		residEvt1.begin();
		for (int b = 0; b < W; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.attnProj[b][d];
		residEvt1.windowSize = W;
		residEvt1.startPosition = startPos;
		residEvt1.dimension = H;
		residEvt1.commit();

		RmsNormEvent normEvt2 = new RmsNormEvent();
		normEvt2.begin();
		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);
		normEvt2.windowSize = W;
		normEvt2.startPosition = startPos;
		normEvt2.dimension = H;
		normEvt2.commit();

		projectWindow(ffnGate[li], ffnGateQ4Dev, ffnGateDev, li, ws.norm2, ws.gate, I, H, startPos);
		projectWindow(ffnUp[li], ffnUpQ4Dev, ffnUpDev, li, ws.norm2, ws.up, I, H, startPos);

		SwiGluEvent swigluEvt = new SwiGluEvent();
		swigluEvt.begin();
		for (int b = 0; b < W; b++)
			for (int i = 0; i < I; i++)
				ws.hidden[b][i] = LlamaTransformerHandler.silu(ws.gate[b][i]) * ws.up[b][i];
		swigluEvt.windowSize = W;
		swigluEvt.startPosition = startPos;
		swigluEvt.dimension = I;
		swigluEvt.commit();

		projectWindow(wDown[li], wDownQ4Dev, wDownDev, li, ws.hidden, ws.ffnOut, H, I, startPos);

		ResidualAddEvent residEvt2 = new ResidualAddEvent();
		residEvt2.begin();
		for (int b = 0; b < W; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.ffnOut[b][d];
		residEvt2.windowSize = W;
		residEvt2.startPosition = startPos;
		residEvt2.dimension = H;
		residEvt2.commit();

		return x;
	}

	/** The per-head Q/K norm and RoPE over a window on the host. */
	private void normalizeAndRotateQk(int li, int startPos, BatchWorkspace ws, int W) {
		normalizeQkHeads(li, startPos, ws, W);
		rotateQk(startPos, ws, W);
	}

	private void normalizeQkHeads(int li, int startPos, BatchWorkspace ws, int W) {
		RmsNormEvent qkNormEvt = new RmsNormEvent();
		qkNormEvt.begin();
		for (int b = 0; b < W; b++) {
			rmsNormPerHead(ws.q[b], qNorm[li], cfg.numHeads(), cfg.headDim(), cfg.rmsNormEps());
			rmsNormPerHead(ws.k[b], kNorm[li], cfg.numKvHeads(), cfg.headDim(), cfg.rmsNormEps());
		}
		qkNormEvt.windowSize = W;
		qkNormEvt.startPosition = startPos;
		qkNormEvt.dimension = cfg.headDim();
		qkNormEvt.commit();
	}

	private void rotateQk(int startPos, BatchWorkspace ws, int W) {
		RopeEvent ropeEvt = new RopeEvent();
		ropeEvt.begin();
		for (int b = 0; b < W; b++) {
			Qwen3Rope.apply(ws.q[b], startPos + b, cfg.numHeads(), cfg.headDim(), cfg.rope());
			Qwen3Rope.apply(ws.k[b], startPos + b, cfg.numKvHeads(), cfg.headDim(), cfg.rope());
		}
		ropeEvt.windowSize = W;
		ropeEvt.startPosition = startPos;
		ropeEvt.dimension = cfg.numHeads() * cfg.headDim() + cfg.numKvHeads() * cfg.headDim();
		ropeEvt.commit();
	}

	/**
	 * The window's KV write and attention, shared by the host and device-region window
	 * paths: host KV first and always, then the device mirror as one copy per tensor, then
	 * attention on the kernel when the mirror is readable and on the CPU otherwise.
	 */
	private void writeKvAndAttend(int startPos, SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			BatchWorkspace ws, DeviceKvCache mirror, int W) {
		GpuAttentionMirror g = gpuAttention;
		WindowStepEvent kvEvt = WindowStepEvent.start();
		for (int b = 0; b < W; b++) {
			kCacheLayer.writeToken(startPos + b, ws.k[b]);
			vCacheLayer.writeToken(startPos + b, ws.v[b]);
		}
		if (g != null)
			mirror = g.appendWindow(mirror, startPos, ws.k, ws.v, W);
		kvEvt.end(WindowStepEvent.KV_WRITE, W, startPos);

		AttentionEvent attnEvt = new AttentionEvent();
		attnEvt.begin();
		if (g == null || !g.attendWindow(mirror, startPos, ws.q, ws.attnOut)) {
			for (int b = 0; b < W; b++) {
				int seqLen = startPos + b + 1;
				float[] kView = kCacheLayer.viewForAttention(seqLen, ws.kDequant);
				float[] vView = vCacheLayer.viewForAttention(seqLen, ws.vDequant);
				gqaInto(cfg, ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
			}
		}
		attnEvt.windowSize = W;
		attnEvt.startPosition = startPos;
		attnEvt.contextLength = startPos + W;
		attnEvt.commit();
	}

	/** Takes a prefill-window region window for {@code W} rows, or null for the host path. */
	private PrefillWindowRegion.Window openPrefillWindow(int W) {
		PrefillWindowRegion region = prefillRegion;
		if (region == null || W <= PrefillWindowRegion.MAX_HOST_WINDOW)
			return null;
		try {
			return region.open(W);
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			warnPrefillRegionFellBackOnce();
			return null;
		}
	}

	/**
	 * {@link #transformerLayerBatch} through the prefill-window device region. The whole
	 * layer runs on the device when attention can (see
	 * {@link PrefillWindowRegion.Window#runLayer}); otherwise the region returns Q, K and V
	 * with each head already normalized (and rotated when RoPE runs on the device), the
	 * KV write and attention run here as on the host path, and the region finishes the
	 * layer. The host KV tensors are written from the region's K and V rows before the
	 * mirror's watermark covers the window, so the mirror stays a copy of them. The
	 * residual stays on the device between layers; the window loop takes it back at the
	 * end. Running out of device memory in the region redoes the layer on the host path
	 * from the layer's input, which the region hands back.
	 */
	private float[][] transformerLayerOnDevice(float[][] x, int li, int startPos, SessionKvTensor kCacheLayer,
			SessionKvTensor vCacheLayer, BatchWorkspace ws, DeviceKvCache mirror, PrefillWindowRegion.Window win) {
		int W = x.length;
		WindowStepEvent devEvt = WindowStepEvent.start();
		boolean whole;
		try {
			whole = win.runLayer(li, x, startPos, mirror, ws.q, ws.k, ws.v);
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			warnPrefillRegionFellBackOnce();
			win.recoverLayerInput(x);
			return transformerLayerBatch(x, li, startPos, kCacheLayer, vCacheLayer, ws, mirror, null);
		}
		devEvt.end(WindowStepEvent.DEVICE_LAYER, W, startPos);

		if (whole) {
			WindowStepEvent kvEvt = WindowStepEvent.start();
			for (int b = 0; b < W; b++) {
				kCacheLayer.writeToken(startPos + b, ws.k[b]);
				vCacheLayer.writeToken(startPos + b, ws.v[b]);
			}
			mirror.markWritten(startPos, W);
			kvEvt.end(WindowStepEvent.KV_WRITE, W, startPos);
			return x;
		}

		if (!prefillRegion.ropeOnDevice())
			rotateQk(startPos, ws, W);
		writeKvAndAttend(startPos, kCacheLayer, vCacheLayer, ws, mirror != null && mirror.live() ? mirror : null, W);

		WindowStepEvent finishEvt = WindowStepEvent.start();
		try {
			win.finishLayer(li, ws.attnOut, x);
		} catch (IllegalStateException ex) {
			if (!GpuLayerOffload.isVramOom(ex))
				throw ex;
			warnPrefillRegionFellBackOnce();
			win.recoverLayerInput(x);
			return transformerLayerBatch(x, li, startPos, kCacheLayer, vCacheLayer, ws, mirror, null);
		}
		finishEvt.end(WindowStepEvent.DEVICE_LAYER, W, startPos);
		return x;
	}

	/**
	 * Takes the window's residual back from the prefill-window region (a no-op when the
	 * host rows are current), inside a {@code juno.WindowStep} {@code device_layer} span
	 * so the download is attributed to the region rather than left outside every span.
	 */
	private static void materializeResidual(PrefillWindowRegion.Window win, float[][] x, int startPos) {
		WindowStepEvent evt = WindowStepEvent.start();
		win.materializeResidual(x);
		evt.end(WindowStepEvent.DEVICE_LAYER, x.length, startPos);
	}

	private void warnPrefillRegionFellBackOnce() {
		if (prefillRegionFallbackWarned.compareAndSet(false, true))
			log.warning("Qwen3: out of device memory in the prefill-window device region - this window's layer runs"
					+ " on the host path between device matmuls. Lower --gpu-layers to leave the region room.");
	}

	private float[][] transformerLayerMultiDecode(float[][] x, int li, int[] positions,
			SessionKvTensor[] kCacheLayers, SessionKvTensor[] vCacheLayers, BatchWorkspace ws,
			DeviceKvCache[] mirrors) {
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

		// Host KV first and always; a device mirror only copies it.
		GpuAttentionMirror g = gpuAttention;
		for (int b = 0; b < N; b++) {
			int pos = positions[b];
			kCacheLayers[b].writeToken(pos, ws.k[b]);
			vCacheLayers[b].writeToken(pos, ws.v[b]);
			if (g != null)
				mirrors[b] = g.append(mirrors[b], pos, ws.k[b], ws.v[b], N);
		}

		if (g == null || !g.attendStreams(mirrors, positions, ws.q, ws.attnOut)) {
			for (int b = 0; b < N; b++) {
				int seqLen = positions[b] + 1;
				float[] kView = kCacheLayers[b].viewForAttention(seqLen, ws.kDequant);
				float[] vView = vCacheLayers[b].viewForAttention(seqLen, ws.vDequant);
				gqaInto(cfg, ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
			}
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

	/** {@link #sgemmLayerInto} inside a {@code juno.WindowStep} projection span, for the prefill window. */
	private void projectWindow(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4,
			DeviceHalfMatrix[] half, int li, float[][] X, float[][] Y, int rows, int cols, int startPos) {
		WindowStepEvent evt = WindowStepEvent.start();
		sgemmLayerInto(quant, q4, half, li, X, Y, rows, cols);
		evt.end(WindowStepEvent.PROJECTION, X.length, startPos);
	}

	private void sgemmLayerInto(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4,
			DeviceHalfMatrix[] half, int li, float[][] X, float[][] Y, int rows, int cols) {
		if (q4 != null && q4[li] != null) {
			backend.sgemmInto(q4[li], X, Y);
			return;
		}
		if (half != null && half[li] != null) {
			backend.sgemmInto(half[li], X, Y);
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
	public void shiftKv(String requestId, int seqLen, int keep, int discard) {
		HandlerContextShift.shiftHost(kvCacheK, kvCacheV, requestId, seqLen, keep, discard,
				Qwen3Rope.shift(cfg.headDim(), cfg.rope()), cfg.numKvHeads());
		GpuAttentionMirror g = gpuAttention;
		if (g != null)
			HandlerContextShift.rewriteMirrors("Qwen3", g.existing(requestId), kvCacheK.get(requestId),
					kvCacheV.get(requestId), seqLen - discard);
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null)
			a.evict(requestId);
	}

	/** Package-private for testing: the request's host K (index 0) and V (index 1) layers, or null. */
	SessionKvTensor[][] hostKv(String requestId) {
		SessionKvTensor[] k = kvCacheK.get(requestId);
		return k == null ? null : new SessionKvTensor[][] { k, kvCacheV.get(requestId) };
	}

	/** Package-private for testing: each layer's device-mirror watermark, -1 where closed; null without mirrors. */
	int[] deviceKvWatermarks(String requestId) {
		GpuAttentionMirror g = gpuAttention;
		return g == null ? null : g.watermarks(requestId);
	}

	/** Package-private for testing: retire the request's device mirrors, so it attends on the host. */
	void retireDeviceKv(String requestId) {
		GpuAttentionMirror g = gpuAttention;
		if (g != null)
			g.retire(requestId);
	}


	@Override
	public void evict(String requestId) {
		SessionKvLayout.releaseLayers(kvCacheK.remove(requestId));
		SessionKvLayout.releaseLayers(kvCacheV.remove(requestId));
		GpuAttentionMirror g = gpuAttention;
		if (g != null)
			g.evict(requestId);
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

		DeviceKvCache[] mirrors = mirrorsFor(requestId);
		ResidentQkvPath region = residentQkv;
		// One device region for the whole token: a layer the region runs whole leaves
		// its output there, and the next layer reads it without an upload.
		try (ResidentQkvPath.Lease lease = region != null ? region.lease() : null) {
			for (int li = 0; li < L; li++)
				x = transformerLayer(x, li, pos, kCache[li], vCache[li], kScratch, vScratch,
						GpuAttentionMirror.layer(mirrors, li, layerOnDevice(li)), region, lease);
		}
		if (region != null && region.ownsResult(x))
			x = x.clone(); // the region's row is overwritten by this thread's next call

		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++)
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch, DeviceKvCache mirror, ResidentQkvPath region,
			ResidentQkvPath.Lease lease) {
		float[] attnProj = null;
		if (region != null && region.eligible(li)) {
			// The device region runs norm, the projections, the per-head norms and RoPE, and
			// with a mirror it can attend through, the KV append, attention and the rest of
			// the layer; its arrays belong to this thread's region until its next call.
			GpuAttentionMirror g = gpuAttention;
			DeviceKvCache regionKv = null;
			if (g != null && region.attendsOnDevice() && mirror != null && mirror.readableThrough(pos)) {
				// Grown here, not inside the region; null once retired for running out of device memory.
				regionKv = g.reserve(mirror, pos);
				mirror = regionKv;
			}
			ResidentQkvPath.Output resident = region.run(li, x, pos, regionKv, lease);
			if (resident != null)
				attnProj = afterRegion(resident, li, pos, kCacheLayer, vCacheLayer, kScratch, vScratch, g, mirror);
			if (resident != null && attnProj == null)
				return resident.layer; // the region ran the rest of the layer as well
		}
		if (attnProj == null) {
			float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());
			attnProj = attentionLayer(new LayerWeights(li), cfg, xNorm, pos, kCacheLayer, vCacheLayer, kScratch,
					vScratch, gpuAttention, mirror);
		}
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = denseFfn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/**
	 * The attention half after a region call: the host KV write, then the output
	 * projection of the region's attention, or the KV append and attention outside the
	 * region when it did not attend. Returns {@code null} when the region ran the whole
	 * layer ({@link ResidentQkvPath.Output#layer} is then the layer output).
	 */
	private float[] afterRegion(ResidentQkvPath.Output resident, int li, int pos, SessionKvTensor kCacheLayer,
			SessionKvTensor vCacheLayer, float[] kScratch, float[] vScratch, GpuAttentionMirror g,
			DeviceKvCache mirror) {
		// Host KV first and always; a device mirror only copies it.
		kCacheLayer.writeToken(pos, resident.k);
		vCacheLayer.writeToken(pos, resident.v);
		LayerWeights w = new LayerWeights(li);
		if (resident.attended) {
			// The region already cast the row into the mirror; it becomes readable now
			// that the host tensors hold it too.
			mirror.markWritten(pos, 1);
			return resident.layerDone ? null : w.matVecWo(resident.attn, cfg.hiddenDim(), cfg.qDim());
		}
		return attendAndProject(w, cfg, resident.q, resident.k, resident.v, pos, kCacheLayer, vCacheLayer, kScratch,
				vScratch, g, mirror);
	}

	/**
	 * Qwen3 attention with per-head Q/K RMS norms — shared by dense and MoE handlers.
	 */
	static float[] attentionLayer(Qwen3AttentionWeights w, Qwen3Config cfg, float[] xNorm, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
		return attentionLayer(w, cfg, xNorm, pos, kCacheLayer, vCacheLayer, kScratch, vScratch, null, null);
	}

	/**
	 * As above, running attention on the GPU kernel when {@code gpu} is non-null and
	 * {@code mirror} holds this position's whole history; the host KV is written
	 * first either way.
	 */
	static float[] attentionLayer(Qwen3AttentionWeights w, Qwen3Config cfg, float[] xNorm, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch, GpuAttentionMirror gpu, DeviceKvCache mirror) {
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
		return attendAndProject(w, cfg, q, k, v, pos, kCacheLayer, vCacheLayer, kScratch, vScratch, gpu, mirror);
	}

	/**
	 * The rest of {@link #attentionLayer} once q, k and v are rotated and the host KV
	 * tensors hold k and v: the device mirror append, attention (on the GPU kernel
	 * when the mirror holds the whole history, on the CPU otherwise) and the output
	 * projection.
	 */
	private static float[] attendAndProject(Qwen3AttentionWeights w, Qwen3Config cfg, float[] q, float[] k,
			float[] v, int pos, SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer, float[] kScratch,
			float[] vScratch, GpuAttentionMirror gpu, DeviceKvCache mirror) {
		int H = cfg.hiddenDim();
		int qDim = cfg.qDim();
		if (gpu != null)
			mirror = gpu.append(mirror, pos, k, v, 1);

		int seqLen = pos + 1;
		float[] attnOut = null;
		if (gpu != null) {
			float[] out = new float[qDim];
			if (gpu.attendOne(mirror, pos, q, out))
				attnOut = out;
		}
		if (attnOut == null) {
			float[] kView = kCacheLayer.viewForAttention(seqLen, kScratch);
			float[] vView = vCacheLayer.viewForAttention(seqLen, vScratch);
			attnOut = gqa(cfg, q, kView, vView, seqLen);
		}
		return w.matVecWo(attnOut, H, qDim);
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
		if (outputProjDev != null) {
			try {
				return backend.sgemv(outputProjDev, xNorm);
			} catch (IllegalStateException ex) {
				matVecFallback.absorb(ex);
			}
		}
		int actualVocab = outputProj.length / cfg.hiddenDim();
		return LlamaTransformerHandler.matVec(outputProj, xNorm, actualVocab, cfg.hiddenDim());
	}

	private float[] matVecLayer(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4, DeviceHalfMatrix[] half,
			int li, float[] x, int rows, int cols) {
		try {
			if (q4 != null && q4[li] != null)
				return backend.sgemv(q4[li], x);
			if (half != null && half[li] != null)
				return backend.sgemv(half[li], x);
		} catch (IllegalStateException ex) {
			matVecFallback.absorb(ex);
		}
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