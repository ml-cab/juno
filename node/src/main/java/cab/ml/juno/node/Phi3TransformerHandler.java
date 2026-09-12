/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.SessionKvLayout;
import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * Phi-3 family transformer forward pass with optional CUDA matmul.
 *
 * <h3>Phi-3 vs LLaMA tensor layout differences</h3>
 * <ol>
 * <li><b>Fused QKV projection</b>: {@code blk.{i}.attn_qkv.weight} shape
 * {@code [H + 2·kvDim, H]}. LLaMA stores separate {@code attn_q/k/v} tensors.
 * This handler keeps the fused tensor in one {@link GgufReader.QuantizedTensor}
 * and uses row-range matVec to extract Q (rows 0..H−1), K (rows H..H+kvDim−1),
 * V (rows H+kvDim..end).
 * <li><b>Fused gate+up FFN</b>: {@code blk.{i}.ffn_up.weight} shape
 * {@code [2·intermediateSize, H]}. Gate occupies rows 0..I−1, up rows I..2I−1.
 * Again kept fused and sliced at call-time.
 * </ol>
 *
 * <h3>Memory layout — why QuantizedTensor and not float[]</h3> Previous
 * versions called {@code GgufReader.tensor()} for every projection weight,
 * which dequantised the entire matrix to float32 eagerly:
 * 
 * <pre>
 *   phi-3.5-mini-instruct.Q4_K_M:
 *     32 layers × 7 projection matrices × avg ~65 MB (float32) ≈ 14.5 GB
 *     --heap 12g  →  OOM  →  Linux SIGKILL  ("Killed", no Java stack trace)
 * </pre>
 * 
 * All seven projection matrices are now stored as
 * {@link GgufReader.QuantizedTensor} (raw Q4_K bytes).
 * {@link LlamaTransformerHandler#matVec(GgufReader.QuantizedTensor, float[], int, int, int)}
 * dequantises one 256-element block at a time during the inner-product loop,
 * keeping the live float footprint at ≈1 kB instead of ≈65 MB per tensor.
 *
 * <p>When constructed with {@link CudaMatVec}, fused QKV and fused gate+up weights
 * are uploaded either as packed Q4_K ({@code --mmq on/auto}) — one physical matrix
 * per fused tensor, GEMV then host-slice — or dequantised and split into FP16
 * {@link DeviceHalfMatrix} slices. Non-fused projections use the same Q4 / FP16
 * choice via {@link MatVec#sgemv}, matching {@link LlamaTransformerHandler} on GPU.
 * 
 * <pre>
 *   Quantised weight memory (Q4_K, 4.5 bits/weight):
 *     32 layers × 7 matrices × avg ~9 MB (Q4_K raw) ≈ 2 GB  ≪  12 GB
 * </pre>
 * 
 * Small tensors (norm weights, token embeddings, output projection) are still
 * loaded as {@code float[]} because they are either tiny or require random
 * row-access patterns that are inconvenient with quantised storage.
 *
 * <h3>Thread safety</h3> Each request uses an isolated KV-cache entry keyed by
 * {@code requestId}. Multiple threads may call {@link #forward} concurrently
 * for distinct requests.
 */
public final class Phi3TransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(Phi3TransformerHandler.class.getName());

	// ── Loaded weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final Phi3RopeConfig ropeCfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	// Small tensors: dequantised to float[] (each is at most a few hundred MB)
	private final float[] tokenEmbd; // [vocabSize × hiddenDim] – first node only
	private final float[] outputNorm; // [hiddenDim] – last node only
	private final float[] outputProj; // [vocabSize × hiddenDim] – last node only

	private final float[][] attnNorm; // [L][hiddenDim]
	private final float[][] ffnNorm; // [L][hiddenDim]

	// Large tensors: kept in quantised form to avoid OOM.
	// Shapes (logical):
	// attnQkv[li] → [H + kvDim + kvDim, H] (fused Q, K, V projections)
	// wo[li] → [H, H] (attention output projection)
	// ffnGateUp[li]→ [2 × intermediateSize, H] (fused gate + up projections)
	// wDown[li] → [H, intermediateSize]
	private final GgufReader.QuantizedTensor[] attnQkv;
	private final GgufReader.QuantizedTensor[] wo;
	private final GgufReader.QuantizedTensor[] ffnGateUp;
	private final GgufReader.QuantizedTensor[] wDown;

	private final MatVec backend;
	/**
	 * Populated when {@link #backend} is {@link CudaMatVec} and upload succeeds;
	 * weights are stored in FP16 on the device (~half the VRAM of FP32). Cleared
	 * after failed upload so forward uses CPU quantised matmul.
	 */
	private DeviceHalfMatrix[] attnQDev = null;
	private DeviceHalfMatrix[] attnKDev = null;
	private DeviceHalfMatrix[] attnVDev = null;
	private DeviceHalfMatrix[] woDev = null;
	private DeviceHalfMatrix[] ffnGateDev = null;
	private DeviceHalfMatrix[] ffnUpDev = null;
	private DeviceHalfMatrix[] wDownDev = null;
	private DeviceHalfMatrix outputProjDev = null;

	/**
	 * Packed Q4_K residency when {@code --mmq} prefers fused GEMV. Fused QKV /
	 * gate_up stay one physical matrix; forward runs one GEMV then host-slices
	 * the output into Q/K/V or gate/up.
	 */
	private DeviceQ4KMatrix[] attnQkvQ4Dev = null;
	private DeviceQ4KMatrix[] woQ4Dev = null;
	private DeviceQ4KMatrix[] ffnGateUpQ4Dev = null;
	private DeviceQ4KMatrix[] wDownQ4Dev = null;

	private final int gpuLayersResolved;

	// Per-request KV cache — lazily allocated and grown on demand.
	// Starts at INITIAL_SEQ_CAPACITY slots, doubles until MAX_SEQ_LEN.
	// Avoids the 554 MB eager pre-allocation that caused node JVM OOM during
	// multi-test runs against phi-3.5-mini with --heap 4g.
	private final Map<String, SessionKvTensor[]> kvCacheK = new ConcurrentHashMap<>();
	private final Map<String, SessionKvTensor[]> kvCacheV = new ConcurrentHashMap<>();
	private final SessionKvLayout kvLayout;

	// ── KV cache adapter (optional — null = dev/stub mode, no eviction) ──────
	private volatile NodeKVCacheAdapter kvAdapter;

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load weights from a Phi-3 GGUF file for the given shard range.
	 *
	 * @param modelPath path to the GGUF file (e.g.
	 *                  phi-3.5-mini-instruct.Q4_K_M.gguf)
	 * @param context   describes which layers/embeddings this node is responsible
	 *                  for
	 */
	public static Phi3TransformerHandler load(Path modelPath, ShardContext context) throws IOException {
		return load(modelPath, context, CpuMatVec.INSTANCE);
	}

	/**
	 * Load a Phi-3 shard with an explicit compute backend.
	 *
	 * <p>{@link CpuMatVec} keeps quantised weights on the host and uses block-wise
	 * dequantisation inside {@link LlamaTransformerHandler#matVec}. {@link CudaMatVec}
	 * dequantises once per matrix, uploads row-split fused slices as FP16 via
	 * {@link CudaMatVec#uploadHalf}, and runs mixed FP16/FP32 {@code cublasSgemmEx}
	 * each forward step.
	 */
	public static Phi3TransformerHandler load(Path modelPath, ShardContext context, MatVec backend)
			throws IOException {
		log.info("Loading Phi-3 GGUF shard: layers " + context.startLayer() + "–" + context.endLayer() + "  embd="
				+ context.hasEmbeddings() + "  outProj=" + context.hasOutputProjection() + "  backend="
				+ backend.getClass().getSimpleName() + "  file=" + modelPath);

		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			log.info("Model: " + cfg);
			return new Phi3TransformerHandler(r, cfg, context, backend);
		}
	}

	private Phi3TransformerHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, MatVec backend) throws IOException {
		this.cfg = cfg;
		this.ropeCfg = Phi3RopeConfig.from(r, cfg);
		this.backend = backend;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();
		this.kvLayout = SessionKvLayout.fromEnv(cfg.kvDim());
		log.info(kvLayout.policySummary());

		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		// ── Small tensors: dequantise eagerly (float[]) ───────────────────────
		// tokenEmbd: [vocabSize × H]. Required for embedding lookup (random row
		// access) — quantised storage would complicate the index calculation.
		// outputProj: [vocabSize × H]. Usually tied to tokenEmbd; either way it
		// is loaded once and the float[] kept for the output matVec.
		// Both fit in a few hundred MB for typical Phi-3 models.
		this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

		attnNorm = new float[L][];
		ffnNorm = new float[L][];

		// ── Large tensors: keep as QuantizedTensor (raw Q4_K bytes) ──────────
		attnQkv = new GgufReader.QuantizedTensor[L];
		wo = new GgufReader.QuantizedTensor[L];
		ffnGateUp = new GgufReader.QuantizedTensor[L];
		wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			log.fine("Loading Phi-3 layer " + i + " weights...");

			// Norm weights are F32 scalars (hiddenDim each) — tiny, keep as float[]
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");

			// Projection tensors: load raw quantised bytes.
			// Logical shapes:
			// attn_qkv.weight : [H + kvDim + kvDim, H]
			// attn_output.weight : [H, H]
			// ffn_up.weight : [2*I, H] (gate rows 0..I-1, up rows I..2I-1)
			// ffn_down.weight : [H, I]
			attnQkv[li] = r.tensorRaw("blk." + i + ".attn_qkv.weight");
			wo[li] = r.tensorRaw("blk." + i + ".attn_output.weight");
			ffnGateUp[li] = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");

			logLayerMemory(i, H, kvDim, I, attnQkv[li], wo[li], ffnGateUp[li], wDown[li]);
		}

		if (backend instanceof GpuMatVec cuda) {
			gpuLayersResolved = uploadGpuWeights(cuda, L, H, kvDim, I);
		} else {
			gpuLayersResolved = 0;
		}
		// CpuMatVec: device fields stay null (defaults above).

		log.info("Phi-3 shard loaded — " + L + " layers, " + (hasEmbeddings ? "with embeddings, " : "")
				+ (hasOutputProj ? "with output projection" : "no output projection"));
	}

	private int uploadGpuWeights(GpuMatVec cuda, int L, int H, int kvDim, int I) {
		GpuLayerOffload policy = GpuLayerOffload.fromEnv();
		int totalLayers = cfg.numLayers();
		boolean tryMmq = MmqOptions.fromEnv().preferMmq() && cuda.supportsQ4KMmq();
		if (tryMmq)
			log.info("Fused Q4_K MMQ enabled (mmq=" + MmqOptions.fromEnv().policyLabel() + ")");
		log.info("Uploading Phi-3 projection weights to GPU (FP16"
				+ (tryMmq ? ", Q4_K packed when available" : "")
				+ ", gpu-layers=" + policy.policyLabel(totalLayers) + ")…");
		DeviceHalfMatrix[] qD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] kD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] vD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] woD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] gD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] uD = new DeviceHalfMatrix[L];
		DeviceHalfMatrix[] dD = new DeviceHalfMatrix[L];
		DeviceQ4KMatrix[] qkvQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] woQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] gateUpQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceQ4KMatrix[] downQ4 = tryMmq ? new DeviceQ4KMatrix[L] : null;
		DeviceHalfMatrix outD = null;
		int resolvedGlobal = 0;
		try {
			for (int li = 0; li < L; li++) {
				int global = startLayer + li;
				if (!policy.isAuto() && !policy.residentForGlobalLayer(global, totalLayers))
					continue;
				try {
					uploadPhi3Layer(cuda, li, H, kvDim, I, tryMmq,
							qD, kD, vD, woD, gD, uD, dD,
							qkvQ4, woQ4, gateUpQ4, downQ4);
					resolvedGlobal = Math.max(resolvedGlobal, global + 1);
				} catch (IllegalStateException ex) {
					if (!phi3HandleLayerOom(policy, ex, global))
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
					log.warning("Phi-3: OOM uploading output projection — using CPU matmul");
				}
			}
			this.attnQDev = qD;
			this.attnKDev = kD;
			this.attnVDev = vD;
			this.woDev = woD;
			this.ffnGateDev = gD;
			this.ffnUpDev = uD;
			this.wDownDev = dD;
			this.attnQkvQ4Dev = qkvQ4;
			this.woQ4Dev = woQ4;
			this.ffnGateUpQ4Dev = gateUpQ4;
			this.wDownQ4Dev = downQ4;
			this.outputProjDev = outD;
			if (policy.isAuto())
				policy = policy.withAutoResolved(resolvedGlobal);
			log.info("Phi-3 GPU weight upload complete (resolved gpu-layers="
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
			Q4KResidentUpload.closeArray(qkvQ4);
			Q4KResidentUpload.closeArray(woQ4);
			Q4KResidentUpload.closeArray(gateUpQ4);
			Q4KResidentUpload.closeArray(downQ4);
			if (outD != null)
				outD.close();
			if (policy.mode() == GpuLayerOffload.Mode.ALL && GpuLayerOffload.isVramOom(ex)) {
				log.warning("Phi-3: not enough GPU VRAM for resident weights (" + ex.getMessage()
						+ "). Using CPU quantised matmul.");
				this.attnQDev = this.attnKDev = this.attnVDev = null;
				this.woDev = this.ffnGateDev = this.ffnUpDev = this.wDownDev = null;
				this.attnQkvQ4Dev = this.woQ4Dev = this.ffnGateUpQ4Dev = this.wDownQ4Dev = null;
				this.outputProjDev = null;
				return 0;
			}
			throw ex;
		}
	}

	private void uploadPhi3Layer(GpuMatVec cuda, int li, int H, int kvDim, int I, boolean tryMmq,
			DeviceHalfMatrix[] qD, DeviceHalfMatrix[] kD, DeviceHalfMatrix[] vD, DeviceHalfMatrix[] woD,
			DeviceHalfMatrix[] gD, DeviceHalfMatrix[] uD, DeviceHalfMatrix[] dD,
			DeviceQ4KMatrix[] qkvQ4, DeviceQ4KMatrix[] woQ4, DeviceQ4KMatrix[] gateUpQ4,
			DeviceQ4KMatrix[] downQ4) {
		if (Q4KResidentUpload.preferPacked(tryMmq, attnQkv[li])) {
			qkvQ4[li] = cuda.uploadKQuant(attnQkv[li].data(), H + 2 * kvDim, H, attnQkv[li].type());
		} else {
			float[] qkvF = LlamaTransformerHandler.dequantize(attnQkv[li], H + 2 * kvDim, H);
			qD[li] = cuda.uploadHalf(rowMajorSlice(qkvF, 0, H, H), H, H);
			kD[li] = cuda.uploadHalf(rowMajorSlice(qkvF, H, kvDim, H), kvDim, H);
			vD[li] = cuda.uploadHalf(rowMajorSlice(qkvF, H + kvDim, kvDim, H), kvDim, H);
		}
		Q4KResidentUpload.uploadInto(cuda, wo[li], H, H, tryMmq, li, woD, woQ4);
		if (Q4KResidentUpload.preferPacked(tryMmq, ffnGateUp[li])) {
			gateUpQ4[li] = cuda.uploadKQuant(ffnGateUp[li].data(), 2 * I, H, ffnGateUp[li].type());
		} else {
			float[] gateUpF = LlamaTransformerHandler.dequantize(ffnGateUp[li], 2 * I, H);
			gD[li] = cuda.uploadHalf(rowMajorSlice(gateUpF, 0, I, H), I, H);
			uD[li] = cuda.uploadHalf(rowMajorSlice(gateUpF, I, I, H), I, H);
		}
		Q4KResidentUpload.uploadInto(cuda, wDown[li], H, I, tryMmq, li, dD, downQ4);
	}

	private boolean phi3HandleLayerOom(GpuLayerOffload policy, IllegalStateException ex, int globalLayer) {
		if (!GpuLayerOffload.isVramOom(ex))
			return false;
		if (policy.mode() == GpuLayerOffload.Mode.ALL)
			return false;
		log.warning("Phi-3: OOM at global layer " + globalLayer + " — partial GPU offload");
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
		Q4KResidentUpload.closeArray(attnQkvQ4Dev);
		Q4KResidentUpload.closeArray(woQ4Dev);
		Q4KResidentUpload.closeArray(ffnGateUpQ4Dev);
		Q4KResidentUpload.closeArray(wDownQ4Dev);
		attnQkvQ4Dev = woQ4Dev = ffnGateUpQ4Dev = wDownQ4Dev = null;
		if (outputProjDev != null && !outputProjDev.isClosed())
			outputProjDev.close();
		outputProjDev = null;
	}

	/** Contiguous row block {@code A[rowStart : rowStart+nRows, 0:cols]} in row-major {@code full}. */
	private static float[] rowMajorSlice(float[] full, int rowStart, int nRows, int cols) {
		float[] out = new float[nRows * cols];
		System.arraycopy(full, rowStart * cols, out, 0, nRows * cols);
		return out;
	}

	/**
	 * Matrix–vector for a row-range of a fused tensor via FP16 slice or CPU.
	 * Prefer {@link #projectFusedQ4} when a packed Q4 fused matrix is resident.
	 */
	private float[] matVecFused(GgufReader.QuantizedTensor quant, DeviceHalfMatrix half,
			float[] x, int rowStart, int rowEnd, int cols) {
		if (half != null)
			return backend.sgemv(half, x);
		return LlamaTransformerHandler.matVec(quant, x, rowStart, rowEnd, cols);
	}

	/** One fused Q4 GEMV then host-slice into {@code outA}/{@code outB} (and optional {@code outC}). */
	private void projectFusedQ4(DeviceQ4KMatrix q4, float[] x, float[] outA, int aLen,
			float[] outB, int bLen, float[] outC, int cLen) {
		float[] full = backend.sgemv(q4, x);
		System.arraycopy(full, 0, outA, 0, aLen);
		System.arraycopy(full, aLen, outB, 0, bLen);
		if (outC != null)
			System.arraycopy(full, aLen + bLen, outC, 0, cLen);
	}

	/** Non-fused projection (wo / down): Q4, FP16, or CPU. */
	private float[] matVecProj(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix q4, DeviceHalfMatrix half,
			float[] x, int rows, int cols) {
		if (q4 != null)
			return backend.sgemv(q4, x);
		if (half != null)
			return backend.sgemv(half, x);
		return LlamaTransformerHandler.matVec(quant, x, 0, rows, cols);
	}

	private static void logLayerMemory(int layer, int H, int kvDim, int I, GgufReader.QuantizedTensor qkv,
			GgufReader.QuantizedTensor wo, GgufReader.QuantizedTensor gateUp, GgufReader.QuantizedTensor down) {
		long quantBytes = (long) qkv.data().length + wo.data().length + gateUp.data().length + down.data().length;
		long float32Bytes = (long) (H + 2L * kvDim) * H * 4 + (long) H * H * 4 + 2L * I * H * 4 + (long) H * I * 4;
		log.fine(String.format("Layer %d: quantised projection weights %.1f MB  (float32 equiv. would be %.1f MB)",
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

		ForwardPassEvent evt = new ForwardPassEvent();
		evt.begin();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		ForwardResult result;
		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			result = ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			result = ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
		}

		evt.handlerType = "phi3";
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

	/**
	 * Batched prefill forward pass for Phi-3 family models. Processes a window of
	 * new prompt tokens in one call, using GEMM for linear projections and the
	 * extended-RoPE per-token (positions differ; cannot be batched).
	 */
	@Override
	public BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
		long start = System.nanoTime();
		int W = request.windowSize();
		int H = cfg.hiddenDim();

		float[][] x;
		if (hasEmbeddings) {
			x = new float[W][H];
			int actualVocab = tokenEmbd.length / H;
			for (int b = 0; b < W; b++) {
				int tokenId = request.tokenIds()[b];
				tokenId = Math.max(0, Math.min(tokenId, actualVocab - 1));
				System.arraycopy(tokenEmbd, tokenId * H, x[b], 0, H);
			}
		} else {
			x = new float[W][H];
			float[] flat = request.activations();
			for (int b = 0; b < W; b++) System.arraycopy(flat, b * H, x[b], 0, H);
		}

		x = runLayersBatch(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			float[] lastX = x[W - 1];
			float[] logits = outputProjection(lastX);
			return new BatchForwardResult(request.requestId(), null, logits, W, System.nanoTime() - start);
		}

		float[] flat = new float[W * H];
		for (int b = 0; b < W; b++) System.arraycopy(x[b], 0, flat, b * H, H);
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

		BatchWorkspace ws = new BatchWorkspace(N, cfg.hiddenDim(), cfg.intermediateSize(),
				kvDim, cfg.numHeads(), maxPos + 1, kvLayout.needsAttentionScratch());

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
				for (int li = 0; li < L; li++) {
					a.flush(requestIds.get(i), startLayer + li, kCaches[i][li], vCaches[i][li], seqLen);
				}
			}
		}

		return x;
	}

	private float[][] transformerLayerMultiDecode(float[][] x, int li, int[] positions,
			SessionKvTensor[] kCacheLayers, SessionKvTensor[] vCacheLayers, BatchWorkspace ws) {
		int N = x.length;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		for (int b = 0; b < N; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);

		sgemmQkvInto(li, ws.norm1, ws.q, ws.k, ws.v, H, kvDim);

		for (int b = 0; b < N; b++) {
			int pos = positions[b];
			Phi3Rope.ropeExt(ws.q[b], pos, cfg.numHeads(), cfg.headDim(), ropeCfg);
			Phi3Rope.ropeExt(ws.k[b], pos, cfg.numKvHeads(), cfg.headDim(), ropeCfg);
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

		sgemmProjInto(wo[li], woQ4Dev, woDev, li, ws.attnOut, ws.attnProj, H, H);

		for (int b = 0; b < N; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.attnProj[b][d];

		for (int b = 0; b < N; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);

		sgemmGateUpInto(li, ws.norm2, ws.gate, ws.up, I, H);

		for (int b = 0; b < N; b++)
			for (int i = 0; i < I; i++)
				ws.hidden[b][i] = LlamaTransformerHandler.silu(ws.gate[b][i]) * ws.up[b][i];

		sgemmProjInto(wDown[li], wDownQ4Dev, wDownDev, li, ws.hidden, ws.ffnOut, H, I);

		for (int b = 0; b < N; b++)
			for (int d = 0; d < H; d++)
				x[b][d] += ws.ffnOut[b][d];

		return x;
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

		BatchWorkspace ws = new BatchWorkspace(W, cfg.hiddenDim(), cfg.intermediateSize(),
				kvDim, cfg.numHeads(), lastPos + 1, kvLayout.needsAttentionScratch());

		for (int li = 0; li < L; li++) {
			x = transformerLayerBatch(x, li, startPos, kCache[li], vCache[li], ws);
		}

		if (a != null) {
			int seqLen = lastPos + 1;
			for (int li = 0; li < L; li++) {
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
			}
		}
		return x;
	}

	/** Pre-allocated workspace for Phi-3 batched-prefill layers. Same purpose as
	 *  LlamaTransformerHandler.BatchWorkspace — eliminates per-layer allocation. */
	private static final class BatchWorkspace {
		final float[][] norm1, norm2, q, k, v, attnOut, attnProj, gate, up, hidden, ffnOut;
		final float[] scores;
		final float[] kDequant, vDequant;

		BatchWorkspace(int W, int H, int I, int kvDim, int numHeads, int maxSeqLen, boolean quantKv) {
			norm1    = new float[W][H];
			norm2    = new float[W][H];
			q        = new float[W][H];
			k        = new float[W][kvDim];
			v        = new float[W][kvDim];
			attnOut  = new float[W][H];
			attnProj = new float[W][H];
			gate     = new float[W][I];
			up       = new float[W][I];
			hidden   = new float[W][I];
			ffnOut   = new float[W][H];
			scores   = new float[numHeads * maxSeqLen];
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
		int W    = x.length;
		int H    = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I    = cfg.intermediateSize();

		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], attnNorm[li], cfg.rmsNormEps(), ws.norm1[b]);

		sgemmQkvInto(li, ws.norm1, ws.q, ws.k, ws.v, H, kvDim);

		for (int b = 0; b < W; b++) {
			Phi3Rope.ropeExt(ws.q[b], startPos + b, cfg.numHeads(), cfg.headDim(), ropeCfg);
			Phi3Rope.ropeExt(ws.k[b], startPos + b, cfg.numKvHeads(), cfg.headDim(), ropeCfg);
		}

		for (int b = 0; b < W; b++) {
			kCacheLayer.writeToken(startPos + b, ws.k[b]);
			vCacheLayer.writeToken(startPos + b, ws.v[b]);
		}

		for (int b = 0; b < W; b++) {
			int seqLen = startPos + b + 1;
			float[] kView = kCacheLayer.viewForAttention(seqLen, ws.kDequant);
			float[] vView = vCacheLayer.viewForAttention(seqLen, ws.vDequant);
			gqaInto(ws.q[b], kView, vView, seqLen, ws.attnOut[b], ws.scores);
		}

		sgemmProjInto(wo[li], woQ4Dev, woDev, li, ws.attnOut, ws.attnProj, H, H);

		for (int b = 0; b < W; b++) for (int d = 0; d < H; d++) x[b][d] += ws.attnProj[b][d];

		for (int b = 0; b < W; b++)
			LlamaTransformerHandler.rmsNormInto(x[b], ffnNorm[li], cfg.rmsNormEps(), ws.norm2[b]);

		sgemmGateUpInto(li, ws.norm2, ws.gate, ws.up, I, H);

		for (int b = 0; b < W; b++)
			for (int i = 0; i < I; i++) ws.hidden[b][i] = LlamaTransformerHandler.silu(ws.gate[b][i]) * ws.up[b][i];

		sgemmProjInto(wDown[li], wDownQ4Dev, wDownDev, li, ws.hidden, ws.ffnOut, H, I);

		for (int b = 0; b < W; b++) for (int d = 0; d < H; d++) x[b][d] += ws.ffnOut[b][d];
		return x;
	}

	/** Zero-allocation gqa: writes into pre-allocated out[] using shared scores scratch. */
	private void gqaInto(float[] q, float[] kCache, float[] vCache, int seqLen,
			float[] out, float[] scores) {
		int H    = cfg.numHeads();
		int Hd   = cfg.headDim();
		int gqaR = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		java.util.Arrays.fill(out, 0f);

		for (int h = 0; h < H; h++) {
			int kvHead = h / gqaR;
			int qBase  = h * Hd;
			int kBase  = kvHead * Hd;

			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * cfg.kvDim() + kBase;
				for (int d = 0; d < Hd; d++) dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}
			LlamaTransformerHandler.softmax(scores, seqLen);

			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOffset = t * cfg.kvDim() + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++) out[outBase + d] += w * vCache[vOffset + d];
			}
		}
	}

	/** Fused QKV: one Q4 GEMM + host slice, else three FP16/CPU row-range GEMMs. */
	private void sgemmQkvInto(int li, float[][] X, float[][] Q, float[][] K, float[][] V, int H, int kvDim) {
		if (attnQkvQ4Dev != null && attnQkvQ4Dev[li] != null) {
			float[][] qkv = backend.sgemm(attnQkvQ4Dev[li], X);
			for (int b = 0; b < X.length; b++) {
				System.arraycopy(qkv[b], 0, Q[b], 0, H);
				System.arraycopy(qkv[b], H, K[b], 0, kvDim);
				System.arraycopy(qkv[b], H + kvDim, V[b], 0, kvDim);
			}
			return;
		}
		sgemmFusedInto(attnQkv[li], attnQDev != null ? attnQDev[li] : null, X, Q, 0, H, H);
		sgemmFusedInto(attnQkv[li], attnKDev != null ? attnKDev[li] : null, X, K, H, H + kvDim, H);
		sgemmFusedInto(attnQkv[li], attnVDev != null ? attnVDev[li] : null, X, V, H + kvDim, H + 2 * kvDim, H);
	}

	/** Fused gate_up: one Q4 GEMM + host slice, else two FP16/CPU row-range GEMMs. */
	private void sgemmGateUpInto(int li, float[][] X, float[][] gate, float[][] up, int I, int H) {
		if (ffnGateUpQ4Dev != null && ffnGateUpQ4Dev[li] != null) {
			float[][] gu = backend.sgemm(ffnGateUpQ4Dev[li], X);
			for (int b = 0; b < X.length; b++) {
				System.arraycopy(gu[b], 0, gate[b], 0, I);
				System.arraycopy(gu[b], I, up[b], 0, I);
			}
			return;
		}
		sgemmFusedInto(ffnGateUp[li], ffnGateDev != null ? ffnGateDev[li] : null, X, gate, 0, I, H);
		sgemmFusedInto(ffnGateUp[li], ffnUpDev != null ? ffnUpDev[li] : null, X, up, I, 2 * I, H);
	}

	private void sgemmProjInto(GgufReader.QuantizedTensor quant, DeviceQ4KMatrix[] q4, DeviceHalfMatrix[] half,
			int li, float[][] X, float[][] Y, int rows, int cols) {
		if (q4 != null && q4[li] != null) {
			float[][] tmp = backend.sgemm(q4[li], X);
			for (int b = 0; b < X.length; b++)
				System.arraycopy(tmp[b], 0, Y[b], 0, rows);
			return;
		}
		sgemmFusedInto(quant, half != null ? half[li] : null, X, Y, 0, rows, cols);
	}

	/** Zero-allocation sgemmFused: writes into pre-allocated rows Y[b]. */
	private void sgemmFusedInto(GgufReader.QuantizedTensor quant, DeviceHalfMatrix half,
			float[][] X, float[][] Y, int rowStart, int rowEnd, int cols) {
		if (half != null) {
			float[][] tmp = backend.sgemm(half, X);
			int rows = rowEnd - rowStart;
			for (int b = 0; b < X.length; b++) System.arraycopy(tmp[b], 0, Y[b], 0, rows);
			return;
		}
		for (int b = 0; b < X.length; b++)
			LlamaTransformerHandler.matVecInto(quant, X[b], Y[b], rowStart, rowEnd, cols);
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
		int L = endLayer - startLayer;
		int kvDim = cfg.kvDim();

		// Lazy initial allocation — 64 slots, grows on demand to avoid OOM.
		// phi-3.5-mini: eager 2048 slots = 554 MB per request; lazy = 17 MB initially.
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
			for (int li = 0; li < L; li++) {
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen);
			}
		}

		return x;
	}

	/**
	 * Wire the {@link NodeKVCacheAdapter} that bridges this handler's in-process
	 * KV arrays with the cluster-level {@link cab.ml.juno.kvcache.KVCacheManager}.
	 *
	 * @param adapter the adapter to use, or {@code null} to disable managed eviction
	 */
	public void setKvAdapter(NodeKVCacheAdapter adapter) {
		this.kvAdapter = adapter;
		if (adapter != null)
			adapter.manager().pagedArena().ifPresent(kvLayout::bindSharedArena);
	}

	/**
	 * Evict all KV state for the given request from the local in-process map
	 * and from the {@link NodeKVCacheAdapter}'s GPU/CPU tiers (if wired).
	 *
	 * @param requestId the request or session identifier
	 */
	@Override
	public void evict(String requestId) {
		SessionKvLayout.releaseLayers(kvCacheK.remove(requestId));
		SessionKvLayout.releaseLayers(kvCacheV.remove(requestId));
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null) {
			a.evict(requestId);
		}
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

	int kvCacheAllocatedSlots(String requestId) {
		SessionKvTensor[] k = kvCacheK.get(requestId);
		return (k == null || k.length == 0) ? 0 : k[0].capacityTokens();
	}

	private float[] transformerLayer(float[] x, int li, int pos,
			SessionKvTensor kCacheLayer, SessionKvTensor vCacheLayer,
			float[] kScratch, float[] vScratch) {
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();

		// ── Attention sub-layer ───────────────────────────────────────────────
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		float[] q;
		float[] k;
		float[] v;
		if (attnQkvQ4Dev != null && attnQkvQ4Dev[li] != null) {
			q = new float[H];
			k = new float[kvDim];
			v = new float[kvDim];
			projectFusedQ4(attnQkvQ4Dev[li], xNorm, q, H, k, kvDim, v, kvDim);
		} else {
			q = matVecFused(attnQkv[li], attnQDev != null ? attnQDev[li] : null, xNorm, 0, H, H);
			k = matVecFused(attnQkv[li], attnKDev != null ? attnKDev[li] : null, xNorm, H, H + kvDim, H);
			v = matVecFused(attnQkv[li], attnVDev != null ? attnVDev[li] : null, xNorm, H + kvDim, H + 2 * kvDim, H);
		}

		Phi3Rope.ropeExt(q, pos, cfg.numHeads(), cfg.headDim(), ropeCfg);
		Phi3Rope.ropeExt(k, pos, cfg.numKvHeads(), cfg.headDim(), ropeCfg);

		kCacheLayer.writeToken(pos, k);
		vCacheLayer.writeToken(pos, v);

		int seqLen = pos + 1;
		float[] kView = kCacheLayer.viewForAttention(seqLen, kScratch);
		float[] vView = vCacheLayer.viewForAttention(seqLen, vScratch);
		float[] attnOut = gqa(q, kView, vView, seqLen);
		float[] attnProj = matVecProj(wo[li],
				woQ4Dev != null ? woQ4Dev[li] : null,
				woDev != null ? woDev[li] : null,
				attnOut, H, H);
		float[] x2 = LlamaTransformerHandler.add(x, attnProj);

		// ── FFN sub-layer ─────────────────────────────────────────────────────
		float[] xNorm2 = LlamaTransformerHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return LlamaTransformerHandler.add(x2, ffnOut);
	}

	/**
	 * SwiGLU: silu(gate(x)) * up(x) → down.
	 *
	 * Gate and up are fused in ffnGateUp; row ranges split them at call-time
	 * (or one Q4 GEMV + host slice when packed residency is active).
	 * ffnGateUp rows: [0, I) → gate projection (SiLU input) [I, 2I) → up projection
	 */
	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate;
		float[] up;
		if (ffnGateUpQ4Dev != null && ffnGateUpQ4Dev[li] != null) {
			gate = new float[I];
			up = new float[I];
			projectFusedQ4(ffnGateUpQ4Dev[li], x, gate, I, up, I, null, 0);
		} else {
			gate = matVecFused(ffnGateUp[li], ffnGateDev != null ? ffnGateDev[li] : null, x, 0, I, H);
			up = matVecFused(ffnGateUp[li], ffnUpDev != null ? ffnUpDev[li] : null, x, I, 2 * I, H);
		}
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = LlamaTransformerHandler.silu(gate[i]) * up[i];
		return matVecProj(wDown[li],
				wDownQ4Dev != null ? wDownQ4Dev[li] : null,
				wDownDev != null ? wDownDev[li] : null,
				hidden, H, I);
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = LlamaTransformerHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		if (outputProjDev != null)
			return backend.sgemv(outputProjDev, xNorm);
		// Use actual tensor dimensions, not cfg.vocabSize(). For phi3,
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
}