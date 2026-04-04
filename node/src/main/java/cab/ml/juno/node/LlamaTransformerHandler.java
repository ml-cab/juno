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
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * LLaMA-family transformer forward pass with a pluggable {@link MatVec}.
 *
 * <p>
 * Handles all LLaMA-compatible architectures: LLaMA 2/3, TinyLlama, Mistral,
 * Gemma — any model whose GGUF {@code general.architecture} is not
 * {@code phi3}. The transformer math (RMS norm, RoPE, GQA, SwiGLU FFN) is
 * identical regardless of the backend. Swapping {@link CpuMatVec} for
 * {@link CudaMatVec} moves all matrix multiplies from CPU threads to
 * cublasSgemv on a GPU.
 *
 * Each node in the cluster owns a contiguous shard of transformer layers.
 * ShardContext tells this handler: hasEmbeddings → run token embedding lookup
 * before layer 0 startLayer..endLayer → which layers to execute
 * hasOutputProjection → run RMS norm + output projection after last layer
 *
 * KV cache: Stored in-process as float[][layer][pos * kvDim + dim]. One entry
 * per (requestId, layer, position). The simple HashMap-based cache here is safe
 * for development; in production the KVCacheManager from the coordinator module
 * handles eviction and GPU offload.
 *
 * Thread safety: Each request uses an isolated KV cache entry keyed by
 * requestId. Multiple threads may call forward() concurrently for distinct
 * requestIds.
 *
 * Math primitives used: - RMS normalisation (rmsNorm) - Matrix–vector multiply
 * (matVec) — A[rows,cols] × x[cols] → y[rows] - Rotary position embeddings
 * (rope) - Grouped-query attention (gqa) - SwiGLU feed-forward network (ffn) -
 * Softmax (inplace, over a slice)
 *
 * All hot primitives are vectorized with the Java Vector API (jdk.incubator.vector).
 * Requires --add-modules jdk.incubator.vector at both compile time and runtime.
 */
public final class LlamaTransformerHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(LlamaTransformerHandler.class.getName());

	// ── Java Vector API — SIMD acceleration ──────────────────────────────────
	// SPECIES_PREFERRED picks the widest available register: AVX-512=16 floats,
	// AVX2=8, SSE=4. All hot loops vectorize to this width automatically.
	private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
	private static final int                  F_LEN     = F_SPECIES.length();

	// Adjacent-pair swap shuffle for RoPE: maps index i to i^1.
	// This swaps (x[0],x[1]), (x[2],x[3]), ... in one rearrange() call.
	// Enables the complex-number rotation x*cos + swap(x)*sin_signed with 3 ops.
	private static final VectorShuffle<Float> ADJACENT_SWAP;
	static {
		int[] perm = new int[F_LEN];
		for (int i = 0; i < F_LEN; i++) perm[i] = i ^ 1;
		ADJACENT_SWAP = VectorShuffle.fromArray(F_SPECIES, perm, 0);
	}

	// Thread-local scratch for RoPE cos/sin (interleaved, length headDim).
	// Avoids per-call allocation while remaining safe across virtual threads
	// (each carrier thread gets its own copy).
	private static final ThreadLocal<float[][]> ROPE_SCRATCH =
		ThreadLocal.withInitial(() -> new float[][] { new float[0], new float[0] });

	// Thread-local scratch for Q8_0 block dequantization (always 32 floats).
	// Eliminates per-row allocation inside the parallel Q8_0 matVec lambda.
	private static final ThreadLocal<float[]> Q8_SCRATCH =
		ThreadLocal.withInitial(() -> new float[32]);

	// Thread-local scratch for F32 raw byte reads (one vector chunk).
	// Avoids fromByteArray — that method compiles unreliably against --release=N
	// ct.sym because jdk.incubator.vector is excluded from cross-compilation headers.
	// Fill F_LEN floats via bit manipulation, then use fromArray (always visible).
	private static final ThreadLocal<float[]> F32_LOAD_SCRATCH =
	    ThreadLocal.withInitial(() -> new float[F_LEN]);

	// ── Loaded weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	// Embedding + output weights (null if this shard doesn't have them)
	private final float[] tokenEmbd; // [vocabSize × hiddenDim] – first node only; kept as float[] for O(1) embedding lookup
	private final float[] outputNorm; // [hiddenDim] – last node only; tiny, kept as float[]
	private final GgufReader.QuantizedTensor outputProj; // [vocabSize × hiddenDim] – last node only; raw bytes, dequantised lazily per-block

	// Per-layer weights stored as raw quantised bytes (dequantised one block at a
	// time inside matVec — never materialised as a full float[] array).
	// This gives 6–8× lower VRAM vs eager float[][], enabling all-layer
	// tensor-parallel loads.
	private final float[][] attnNorm; // [L][hiddenDim] — tiny F32 scalars, kept as float[]
	private final float[][] ffnNorm;  // [L][hiddenDim] — tiny F32 scalars, kept as float[]
	private final GgufReader.QuantizedTensor[] wq;    // [L][hiddenDim × hiddenDim]
	private final GgufReader.QuantizedTensor[] wk;    // [L][kvDim × hiddenDim]
	private final GgufReader.QuantizedTensor[] wv;    // [L][kvDim × hiddenDim]
	private final GgufReader.QuantizedTensor[] wo;    // [L][hiddenDim × hiddenDim]
	private final GgufReader.QuantizedTensor[] wGate; // [L][intermediateSize × hiddenDim]
	private final GgufReader.QuantizedTensor[] wUp;   // [L][intermediateSize × hiddenDim]
	private final GgufReader.QuantizedTensor[] wDown; // [L][hiddenDim × intermediateSize]

	// Per-request KV cache — lazily allocated and grown on demand.
	// Starts at INITIAL_SEQ_CAPACITY slots, doubles until MAX_SEQ_LEN.
	private final Map<String, float[][]> kvCacheK = new HashMap<>();
	private final Map<String, float[][]> kvCacheV = new HashMap<>();
	private static final int MAX_SEQ_LEN        = 2048;
	private static final int INITIAL_SEQ_CAPACITY = 64; // grows on demand

	// ── MatVec backend (CPU or CUDA) ─────────────────────────────────────────
	private final MatVec backend;

	// ── Device-resident weight matrices (non-null only when backend is CudaMatVec) ──
	// Weights are dequantized to float32 once at load time and uploaded to GPU
	// so every forward pass avoids per-call H2D weight transfers.
	// null on all arrays when backend == CpuMatVec (default CPU path).
	private final DeviceFloatMatrix[] wqDev;
	private final DeviceFloatMatrix[] wkDev;
	private final DeviceFloatMatrix[] wvDev;
	private final DeviceFloatMatrix[] woDev;
	private final DeviceFloatMatrix[] wGateDev;
	private final DeviceFloatMatrix[] wUpDev;
	private final DeviceFloatMatrix[] wDownDev;
	private final DeviceFloatMatrix outputProjDev;

	// ── KV cache adapter (optional — null = dev/stub mode, no eviction) ──────
	private volatile NodeKVCacheAdapter kvAdapter;

	// ── Factory ───────────────────────────────────────────────────────────────

	public static LlamaTransformerHandler load(Path modelPath, ShardContext context) throws IOException {
		log.info("Loading GGUF shard: layers " + context.startLayer() + "–" + context.endLayer() + "  embd="
				+ context.hasEmbeddings() + "  outProj=" + context.hasOutputProjection() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			log.info("Model: " + cfg);
			return new LlamaTransformerHandler(r, cfg, context, CpuMatVec.INSTANCE);
		}
	}

	public static LlamaTransformerHandler load(Path modelPath, ShardContext context, MatVec backend)
			throws IOException {
		log.info("Loading GGUF shard: layers " + context.startLayer() + "–" + context.endLayer() + "  embd="
				+ context.hasEmbeddings() + "  outProj=" + context.hasOutputProjection() + "  backend="
				+ backend.getClass().getSimpleName() + "  file=" + modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			log.info("Model: " + cfg);
			return new LlamaTransformerHandler(r, cfg, context, backend);
		}
	}

	/** Direct constructor used by {@link #newTestInstance} — no GGUF I/O. */
	@SuppressWarnings("java:S107")
	private LlamaTransformerHandler(
			LlamaConfig cfg, int startLayer, int endLayer,
			boolean hasEmbeddings, boolean hasOutputProj,
			float[] tokenEmbd, float[] outputNorm,
			GgufReader.QuantizedTensor outputProj,
			float[][] attnNorm, float[][] ffnNorm,
			GgufReader.QuantizedTensor[] wq,
			GgufReader.QuantizedTensor[] wk,
			GgufReader.QuantizedTensor[] wv,
			GgufReader.QuantizedTensor[] wo,
			GgufReader.QuantizedTensor[] wGate,
			GgufReader.QuantizedTensor[] wUp,
			GgufReader.QuantizedTensor[] wDown,
			MatVec backend) {
		this.cfg           = cfg;
		this.backend       = backend;
		this.startLayer    = startLayer;
		this.endLayer      = endLayer;
		this.hasEmbeddings = hasEmbeddings;
		this.hasOutputProj = hasOutputProj;
		this.tokenEmbd    = tokenEmbd;
		this.outputNorm   = outputNorm;
		this.outputProj   = outputProj;
		this.attnNorm     = attnNorm;
		this.ffnNorm      = ffnNorm;
		this.wq           = wq;
		this.wk           = wk;
		this.wv           = wv;
		this.wo           = wo;
		this.wGate        = wGate;
		this.wUp          = wUp;
		this.wDown        = wDown;
		// Direct (test) constructor: no GPU upload — device matrices are unused.
		this.wqDev = this.wkDev = this.wvDev = this.woDev =
				this.wGateDev = this.wUpDev = this.wDownDev = null;
		this.outputProjDev = null;
	}

	private LlamaTransformerHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, MatVec backend)
			throws IOException {
		this.cfg           = cfg;
		this.backend       = backend;
		this.startLayer    = ctx.startLayer();
		this.endLayer      = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L = endLayer - startLayer;

		this.tokenEmbd  = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj  ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj  ? loadOutputProjection(r) : null;

		attnNorm = new float[L][];
		ffnNorm  = new float[L][];
		wq    = new GgufReader.QuantizedTensor[L];
		wk    = new GgufReader.QuantizedTensor[L];
		wv    = new GgufReader.QuantizedTensor[L];
		wo    = new GgufReader.QuantizedTensor[L];
		wGate = new GgufReader.QuantizedTensor[L];
		wUp   = new GgufReader.QuantizedTensor[L];
		wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			log.fine("Loading layer " + i + " weights...");
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li]  = r.tensor("blk." + i + ".ffn_norm.weight");
			wq[li]    = r.tensorRaw("blk." + i + ".attn_q.weight");
			wk[li]    = r.tensorRaw("blk." + i + ".attn_k.weight");
			wv[li]    = r.tensorRaw("blk." + i + ".attn_v.weight");
			wo[li]    = r.tensorRaw("blk." + i + ".attn_output.weight");
			wGate[li] = r.tensorRaw("blk." + i + ".ffn_gate.weight");
			wUp[li]   = r.tensorRaw("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensorRaw("blk." + i + ".ffn_down.weight");
		}

		log.info("Shard loaded — " + L + " layers, " + (hasEmbeddings ? "with embeddings, " : "")
				+ (hasOutputProj ? "with output projection" : "no output projection"));

		// Upload dequantized weights to GPU when a CudaMatVec backend is provided.
		// This happens once at load time so forward passes avoid per-call H2D transfers.
		if (backend instanceof CudaMatVec cuda) {
			log.info("Uploading dequantized weights to GPU (cuda-resident)...");
			int H  = cfg.hiddenDim();
			int KV = cfg.kvDim();
			int I  = cfg.intermediateSize();
			int V  = cfg.vocabSize();
			wqDev    = new DeviceFloatMatrix[L];
			wkDev    = new DeviceFloatMatrix[L];
			wvDev    = new DeviceFloatMatrix[L];
			woDev    = new DeviceFloatMatrix[L];
			wGateDev = new DeviceFloatMatrix[L];
			wUpDev   = new DeviceFloatMatrix[L];
			wDownDev = new DeviceFloatMatrix[L];
			for (int li = 0; li < L; li++) {
				wqDev[li]    = cuda.upload(dequantize(wq[li],   H,  H), H,  H);
				wkDev[li]    = cuda.upload(dequantize(wk[li],   KV, H), KV, H);
				wvDev[li]    = cuda.upload(dequantize(wv[li],   KV, H), KV, H);
				woDev[li]    = cuda.upload(dequantize(wo[li],   H,  H), H,  H);
				wGateDev[li] = cuda.upload(dequantize(wGate[li], I, H), I,  H);
				wUpDev[li]   = cuda.upload(dequantize(wUp[li],   I, H), I,  H);
				wDownDev[li] = cuda.upload(dequantize(wDown[li], H, I), H,  I);
			}
			outputProjDev = (outputProj != null)
					? cuda.upload(dequantize(outputProj, V, H), V, H)
					: null;
			log.info("GPU weight upload complete.");
		} else {
			wqDev = wkDev = wvDev = woDev = wGateDev = wUpDev = wDownDev = null;
			outputProjDev = null;
		}
	}

	private static GgufReader.QuantizedTensor loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight")) {
			return r.tensorRaw("output.weight");
		}
		log.info("output.weight not found — model uses tied embeddings; reusing token_embd.weight as output projection");
		return r.tensorRaw("token_embd.weight");
	}

	// ── KV adapter wiring ─────────────────────────────────────────────────────

	public void setKvAdapter(NodeKVCacheAdapter adapter) {
		this.kvAdapter = adapter;
	}

	public void evict(String requestId) {
		kvCacheK.remove(requestId);
		kvCacheV.remove(requestId);
		NodeKVCacheAdapter a = kvAdapter;
		if (a != null) a.evict(requestId);
	}

	static LlamaTransformerHandler newTestInstance(
			int vocabSize, int hiddenDim, int numHeads, int numKvHeads,
			int numLayers, int startLayer, int endLayer,
			boolean hasEmbd, boolean hasOutProj,
			NodeKVCacheAdapter adapter) {

		LlamaConfig cfg = LlamaConfig.synthetic(vocabSize, hiddenDim, numHeads, numKvHeads, numLayers);

		int L    = endLayer - startLayer;
		int H    = hiddenDim;
		int kvDim = (hiddenDim / numHeads) * numKvHeads;
		int I    = hiddenDim * 4;

		java.util.Random rng = new java.util.Random(42);

		float[] tokenEmbd  = hasEmbd    ? randF32(vocabSize * H, rng) : null;
		float[] outputNorm = hasOutProj ? randF32(H, rng)             : null;
		GgufReader.QuantizedTensor outputProj =
				hasOutProj ? f32Tensor("output.weight", vocabSize, H, rng) : null;

		float[][] attnNorm = new float[L][];
		float[][] ffnNorm  = new float[L][];
		GgufReader.QuantizedTensor[] wq    = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wk    = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wv    = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wo    = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wGate = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wUp   = new GgufReader.QuantizedTensor[L];
		GgufReader.QuantizedTensor[] wDown = new GgufReader.QuantizedTensor[L];

		for (int li = 0; li < L; li++) {
			attnNorm[li] = randF32(H, rng);
			ffnNorm[li]  = randF32(H, rng);
			wq[li]    = f32Tensor("wq."    + li, H,     H,    rng);
			wk[li]    = f32Tensor("wk."    + li, kvDim, H,    rng);
			wv[li]    = f32Tensor("wv."    + li, kvDim, H,    rng);
			wo[li]    = f32Tensor("wo."    + li, H,     H,    rng);
			wGate[li] = f32Tensor("wGate." + li, I,     H,    rng);
			wUp[li]   = f32Tensor("wUp."   + li, I,     H,    rng);
			wDown[li] = f32Tensor("wDown." + li, H,     I,    rng);
		}

		LlamaTransformerHandler h = new LlamaTransformerHandler(
				cfg, startLayer, endLayer, hasEmbd, hasOutProj,
				tokenEmbd, outputNorm, outputProj,
				attnNorm, ffnNorm, wq, wk, wv, wo, wGate, wUp, wDown,
				CpuMatVec.INSTANCE);
		h.kvAdapter = adapter;
		return h;
	}

	private static float[] randF32(int n, java.util.Random rng) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++) a[i] = (rng.nextFloat() - 0.5f) * 0.02f;
		return a;
	}

	private static GgufReader.QuantizedTensor f32Tensor(String name, int rows, int cols, java.util.Random rng) {
		int n = rows * cols;
		java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++) bb.putFloat((rng.nextFloat() - 0.5f) * 0.02f);
		return new GgufReader.QuantizedTensor(name, 0, n, bb.array());
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

		evt.handlerType         = "llama";
		evt.requestId           = request.requestId();
		evt.startPosition       = request.startPosition();
		evt.layerCount          = endLayer - startLayer;
		evt.hasOutputProjection = hasOutputProj;
		evt.commit();

		return result;
	}

	@Override
	public boolean isReady() { return true; }

	// ── Transformer forward pass ──────────────────────────────────────────────

	private float[] getInitialActivation(ForwardRequest request) {
		if (hasEmbeddings) {
			int[] tokenIds = request.tokenIds();
			int tokenId = tokenIds[tokenIds.length - 1];
			tokenId = Math.max(0, Math.min(tokenId, cfg.vocabSize() - 1));
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
		int L     = endLayer - startLayer;
		int kvDim = cfg.kvDim();

		boolean isNew = !kvCacheK.containsKey(requestId);
		kvCacheK.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);
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

		for (int li = 0; li < L; li++) {
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);
		}

		if (a != null) {
			int seqLen = pos + 1;
			for (int li = 0; li < L; li++) {
				a.flush(requestId, startLayer + li, kCache[li], vCache[li], seqLen, kvDim);
			}
		}

		return x;
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

	int kvCacheAllocatedSlots(String requestId) {
		float[][] k = kvCacheK.get(requestId);
		return (k == null || k.length == 0) ? 0 : k[0].length / cfg.kvDim();
	}

	private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();

		// ── Attention sub-layer ───────────────────────────────────────────────
		float[] xNorm = rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		// Project to Q, K, V
		float[] q = matVecLayer(wq[li], wqDev != null ? wqDev[li] : null, xNorm, H, H);
		float[] k = matVecLayer(wk[li], wkDev != null ? wkDev[li] : null, xNorm, cfg.kvDim(), H);
		float[] v = matVecLayer(wv[li], wvDev != null ? wvDev[li] : null, xNorm, cfg.kvDim(), H);

		rope(q, pos, cfg.numHeads(),   cfg.headDim(), cfg.ropeTheta());
		rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * cfg.kvDim(), cfg.kvDim());
		System.arraycopy(v, 0, vCacheLayer, pos * cfg.kvDim(), cfg.kvDim());

		// Grouped-query attention
		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);

		// Output projection + residual
		float[] attnProj = matVecLayer(wo[li], woDev != null ? woDev[li] : null, attnOut, H, H);
		float[] x2 = add(x, attnProj);

		// ── FFN sub-layer ─────────────────────────────────────────────────────
		float[] xNorm2 = rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return add(x2, ffnOut);
	}

	/**
	 * SwiGLU feed-forward: silu(gate(x)) * up(x) → down(result).
	 *
	 * <p>Allocations eliminated vs original:
	 * <ul>
	 *   <li>silu applied in-place on gate[] (reuses the array)
	 *   <li>gate*up written into up[] (eliminates the hidden[intermediateSize] allocation)
	 * </ul>
	 * For TinyLlama (intermediateSize=5632) this saves ~22 KB per ffn call,
	 * ~484 KB per decode step across 22 layers.
	 */
	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = matVecLayer(wGate[li], wGateDev != null ? wGateDev[li] : null, x, I, H);
		float[] up = matVecLayer(wUp[li], wUpDev != null ? wUpDev[li] : null, x, I, H);
		// SiLU(gate) * up
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = silu(gate[i]) * up[i];
		return matVecLayer(wDown[li], wDownDev != null ? wDownDev[li] : null, hidden, H, I);
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = rmsNorm(x, outputNorm, cfg.rmsNormEps());
		return outputProjDev != null
			? backend.sgemv(outputProjDev, xNorm)
			: matVec(outputProj, xNorm, cfg.vocabSize(), cfg.hiddenDim());
	}

	// ── Math primitives ───────────────────────────────────────────────────────

	/**
	 * RMS normalisation: x_norm[i] = w[i] * x[i] / rms(x)
	 *
	 * <p>Vectorized with two passes:
	 * <ol>
	 *   <li>Sum of squares via FMA (v·v + acc) — maps to VFMADD on AVX2/AVX-512
	 *   <li>Element-wise w[i]*x[i]*scale — 2 vmulps per chunk
	 * </ol>
	 * On AVX2 (8-wide): processes 8 floats per iteration in both passes.
	 */
	static float[] rmsNorm(float[] x, float[] w, float eps) {
		int n     = x.length;
		int limit = F_SPECIES.loopBound(n);

		// Pass 1: vectorized sum of squares
		var ssVec = FloatVector.zero(F_SPECIES);
		int i = 0;
		for (; i < limit; i += F_LEN) {
			var v = FloatVector.fromArray(F_SPECIES, x, i);
			ssVec = v.fma(v, ssVec); // ssVec += v*v
		}
		float ss = ssVec.reduceLanes(VectorOperators.ADD);
		for (; i < n; i++) ss += x[i] * x[i];

		float scale  = 1f / (float) Math.sqrt(ss / n + eps);
		float[] out  = new float[n];
		var vscale   = FloatVector.broadcast(F_SPECIES, scale);

		// Pass 2: vectorized w[i] * x[i] * scale
		i = 0;
		for (; i < limit; i += F_LEN) {
			FloatVector.fromArray(F_SPECIES, w, i)
			           .mul(FloatVector.fromArray(F_SPECIES, x, i))
			           .mul(vscale)
			           .intoArray(out, i);
		}
		for (; i < n; i++) out[i] = w[i] * x[i] * scale;
		return out;
	}

	// ── Backend dispatch ─────────────────────────────────────────────────────

	/**
	 * Route a matrix-vector multiply through the correct backend.
	 *
	 * <p>When {@code dev} is non-null (i.e. the backend is {@link CudaMatVec} and
	 * the weights were uploaded at load time) the call goes through
	 * {@link MatVec#sgemv(DeviceFloatMatrix, float[])} which uses the
	 * device-resident weight matrix — no H2D transfer per call.
	 *
	 * <p>When {@code dev} is null (CPU path) the call falls through to the static
	 * quantized matVec, which is the original behaviour.
	 *
	 * @param quant the quantized weight tensor (used on CPU path)
	 * @param dev   the device-resident dequantized version, or {@code null} for CPU
	 * @param x     input vector
	 * @param rows  output dimension
	 * @param cols  input dimension
	 */
	private float[] matVecLayer(GgufReader.QuantizedTensor quant, DeviceFloatMatrix dev,
			float[] x, int rows, int cols) {
		if (dev != null) {
			return backend.sgemv(dev, x);
		}
		return matVec(quant, x, rows, cols);
	}

	/**
	 * Dequantize a {@link GgufReader.QuantizedTensor} to a flat float[] array.
	 *
	 * <p>Used once per weight matrix at load time when {@link CudaMatVec} is the
	 * backend. The returned array is immediately uploaded via
	 * {@link CudaMatVec#upload} and then eligible for GC — it is never kept alive.
	 *
	 * @param t    quantized weight tensor
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return row-major float[rows * cols] with all values dequantized
	 */
	static float[] dequantize(GgufReader.QuantizedTensor t, int rows, int cols) {
		return switch (t.type()) {
		case 0  -> dequantizeF32(t.data(), rows, cols);
		case 8  -> dequantizeQ8_0(t.data(), rows, cols);
		case 12 -> dequantizeQ4K(t.data(), rows, cols);
		case 13 -> dequantizeQ5K(t.data(), rows, cols);
		case 14 -> dequantizeQ6K(t.data(), rows, cols);
		default -> throw new UnsupportedOperationException(
				"dequantize not implemented for GGML type " + t.type());
		};
	}

	private static float[] dequantizeF32(byte[] raw, int rows, int cols) {
		int n = rows * cols;
		float[] out = new float[n];
		java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; i++) out[i] = bb.getFloat(i * 4);
		return out;
	}

	private static float[] dequantizeQ8_0(byte[] raw, int rows, int cols) {
		final int BLOCK_SIZE = 32;
		final int BLOCK_BYTES = 34;
		int n = rows * cols;
		float[] out = new float[n];
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowOff = r * bytesPerRow;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowOff + b * BLOCK_BYTES;
				float scale = GgufReader.f16ToF32(readLE16(raw, bo));
				int outBase = r * cols + b * BLOCK_SIZE;
				for (int i = 0; i < BLOCK_SIZE; i++)
					out[outBase + i] = scale * raw[bo + 2 + i];
			}
		}
		return out;
	}

	private static float[] dequantizeQ4K(byte[] raw, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 144;
		int n = rows * cols;
		float[] out = new float[n];
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo     = rowByteOff + b * BLOCK_BYTES;
				int scBase = bo + 4;
				int qsBase = bo + 16;
				float d    = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < BLOCK_SIZE; g += 64) {
					int s0 = g / 32, s1 = s0 + 1;
					float scale0 = d    * q4kScaleRaw(raw, scBase, s0);
					float min0   = dmin * q4kMinRaw  (raw, scBase, s0);
					float scale1 = d    * q4kScaleRaw(raw, scBase, s1);
					float min1   = dmin * q4kMinRaw  (raw, scBase, s1);
					int outOff = xBase + b * BLOCK_SIZE + g;
					for (int i = 0; i < 32; i++)
						out[outOff + i]      = scale0 * (raw[qsBase + qi + i] & 0x0F) - min0;
					for (int i = 0; i < 32; i++)
						out[outOff + 32 + i] = scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1;
					qi += 32;
				}
			}
		}
		return out;
	}

	private static float[] dequantizeQ5K(byte[] raw, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 176;
		int n = rows * cols;
		float[] out = new float[n];
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo     = rowByteOff + b * BLOCK_BYTES;
				int scBase = bo + 4;
				int qhBase = bo + 16;
				int qsBase = bo + 48;
				float d    = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				int qi = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = g * 2, s1 = s0 + 1;
					float scale0 = d    * q4kScaleRaw(raw, scBase, s0);
					float min0   = dmin * q4kMinRaw  (raw, scBase, s0);
					float scale1 = d    * q4kScaleRaw(raw, scBase, s1);
					float min1   = dmin * q4kMinRaw  (raw, scBase, s1);
					int hiBit0 = g * 2, hiBit1 = g * 2 + 1;
					int outOff = xBase + b * BLOCK_SIZE + g * 64;
					for (int l = 0; l < 32; l++) {
						int lo = raw[qsBase + qi + l] & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit0) & 1;
						out[outOff + l] = scale0 * (lo | (hi << 4)) - min0;
					}
					for (int l = 0; l < 32; l++) {
						int lo = (raw[qsBase + qi + l] >>> 4) & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit1) & 1;
						out[outOff + 32 + l] = scale1 * (lo | (hi << 4)) - min1;
					}
					qi += 32;
				}
			}
		}
		return out;
	}

	private static float[] dequantizeQ6K(byte[] raw, int rows, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 210;
		int n = rows * cols;
		float[] out = new float[n];
		int blocksPerRow = cols / BLOCK_SIZE;
		int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		for (int r = 0; r < rows; r++) {
			int rowByteOff = r * bytesPerRow;
			int xBase = r * cols;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo  = rowByteOff + b * BLOCK_BYTES;
				float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));
				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff  = xBase + b * BLOCK_SIZE + half * 128;
					for (int l = 0; l < 32; l++) {
						int is   = l / 16;
						int qlL  = raw[qlOff + l]      & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL  = raw[qhOff + l]      & 0xFF;
						int q1 = ((qlL  & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL  >> 4)   | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4)   | (((qhL >> 6) & 3) << 4)) - 32;
						float d1 = d * raw[scOff + is];
						float d2 = d * raw[scOff + is + 2];
						float d3 = d * raw[scOff + is + 4];
						float d4 = d * raw[scOff + is + 6];
						out[xOff + l]       = d1 * q1;
						out[xOff + l + 32]  = d2 * q2;
						out[xOff + l + 64]  = d3 * q3;
						out[xOff + l + 96]  = d4 * q4;
					}
				}
			}
		}
		return out;
	}

	/**
	 * Float[] matrix–vector multiply: y[rows] = A[rows, cols] × x[cols].
	 *
	 * <p>Inner dot-product vectorized with FMA. Outer loop parallelized via
	 * ForkJoinPool for rows ≥ 256 (parallel threshold unchanged).
	 *
	 * <p>Each row's dot-product: {@code acc += A[base+c] * x[c]} vectorizes to
	 * a single VFMADD instruction per F_LEN elements on AVX2/AVX-512.
	 */
	static float[] matVec(float[] A, float[] x, int rows, int cols) {
		float[] y     = new float[rows];
		int limit     = F_SPECIES.loopBound(cols);
		if (rows >= 256) {
			java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
				int base = r * cols;
				var acc  = FloatVector.zero(F_SPECIES);
				int c    = 0;
				for (; c < limit; c += F_LEN) {
					acc = FloatVector.fromArray(F_SPECIES, A, base + c)
					                 .fma(FloatVector.fromArray(F_SPECIES, x, c), acc);
				}
				float sum = acc.reduceLanes(VectorOperators.ADD);
				for (; c < cols; c++) sum += A[base + c] * x[c];
				y[r] = sum;
			});
		} else {
			for (int r = 0; r < rows; r++) {
				int base = r * cols;
				var acc  = FloatVector.zero(F_SPECIES);
				int c    = 0;
				for (; c < limit; c += F_LEN) {
					acc = FloatVector.fromArray(F_SPECIES, A, base + c)
					                 .fma(FloatVector.fromArray(F_SPECIES, x, c), acc);
				}
				float sum = acc.reduceLanes(VectorOperators.ADD);
				for (; c < cols; c++) sum += A[base + c] * x[c];
				y[r] = sum;
			}
		}
		return y;
	}

	/**
	 * Rotary position embeddings (RoPE). Applied in-place to x[nHeads * headDim].
	 *
	 * <p><b>Optimizations vs original:</b>
	 * <ol>
	 *   <li>Cos/sin precomputed ONCE for all headDim/2 frequencies, then reused
	 *       across all nHeads. For TinyLlama (32 Q-heads): saves 31× Math.pow +
	 *       cos + sin calls per rope() invocation — from 1024 to 32 trig ops.
	 *   <li>Cos/sin stored interleaved as cos[2i]=cos[2i+1]=c and
	 *       sin[2i]=-s, sin[2i+1]=+s, enabling a 3-op vector rotation
	 *       (load, swap-adjacent-pairs, FMA) per chunk.
	 *   <li>Thread-local scratch avoids per-call allocation of two float[headDim]
	 *       arrays (64 floats = 256 bytes for TinyLlama headDim=64).
	 * </ol>
	 *
	 * <p>Vector rotation identity (adjacent-pair convention):
	 * <pre>
	 *   new_x[2i]   = x[2i]*cos[i] + swap(x)[2i]  * sin[2i]   (sin[2i]=-s,   result = x[2i]*c - x[2i+1]*s)
	 *   new_x[2i+1] = x[2i+1]*cos[i] + swap(x)[2i+1] * sin[2i+1] (sin[2i+1]=+s, result = x[2i+1]*c + x[2i]*s)
	 * </pre>
	 */
	static void rope(float[] x, int pos, int nHeads, int headDim, float ropeTheta) {
		int half = headDim / 2;

		// Get or grow per-thread cos/sin scratch (interleaved, length headDim)
		float[][] scratch = ROPE_SCRATCH.get();
		if (scratch[0].length < headDim) {
			scratch[0] = new float[headDim];
			scratch[1] = new float[headDim];
		}
		float[] cosArr = scratch[0];
		float[] sinArr = scratch[1];

		// Precompute interleaved cos/sin once — reused for all nHeads
		for (int i = 0; i < half; i++) {
			double freq  = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
			double angle = pos * freq;
			float  c     = (float) Math.cos(angle);
			float  s     = (float) Math.sin(angle);
			cosArr[2 * i]     =  c;  // even position: rotate with +cos
			cosArr[2 * i + 1] =  c;  // odd  position: rotate with +cos
			sinArr[2 * i]     = -s;  // even position: subtract x[2i+1]*sin
			sinArr[2 * i + 1] = +s;  // odd  position: add      x[2i]*sin
		}

		// Apply rotation to each head using vector ops
		int limit = F_SPECIES.loopBound(headDim);
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			int d    = 0;
			for (; d < limit; d += F_LEN) {
				var vx    = FloatVector.fromArray(F_SPECIES, x,      base + d);
				var vcos  = FloatVector.fromArray(F_SPECIES, cosArr, d);
				var vsin  = FloatVector.fromArray(F_SPECIES, sinArr, d);
				// vswap: each adjacent pair (x[2i], x[2i+1]) becomes (x[2i+1], x[2i])
				var vswap = vx.rearrange(ADJACENT_SWAP);
				// result = vx*cos + vswap*sin  (sin carries the sign encoding above)
				vx.mul(vcos).add(vswap.mul(vsin)).intoArray(x, base + d);
			}
			// Scalar tail — runs only if headDim not a multiple of F_LEN
			for (; d < headDim; d += 2) {
				float c  = cosArr[d];
				float s  = sinArr[d + 1]; // +s stored at odd index
				float x0 = x[base + d], x1 = x[base + d + 1];
				x[base + d]     = x0 * c - x1 * s;
				x[base + d + 1] = x0 * s + x1 * c;
			}
		}
	}

	/**
	 * Grouped-query attention (GQA).
	 *
	 * <p>Both inner loops are vectorized:
	 * <ul>
	 *   <li>Q·K dot product: FMA over headDim floats per (head, time) pair
	 *   <li>V weighted sum: broadcast(score) then FMA over headDim floats per (head, time) pair
	 * </ul>
	 * For TinyLlama (H=32, Hd=64, seqLen≈50): ~32*50*64 = 102K FMA ops per decode step,
	 * vectorized to ~12.8K vector instructions on AVX2.
	 */
	private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
		int   H    = cfg.numHeads();
		int   Hd   = cfg.headDim();
		int   gqa  = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));

		float[] out    = new float[H * Hd];
		float[] scores = new float[seqLen];
		int     hdLimit = F_SPECIES.loopBound(Hd);

		for (int h = 0; h < H; h++) {
			int kvHead  = h / gqa;
			int qBase   = h * Hd;
			int kBase   = kvHead * Hd;

			// ── Q·K dot products — vectorized over headDim ────────────────────
			for (int t = 0; t < seqLen; t++) {
				int kOffset = t * cfg.kvDim() + kBase;
				var acc     = FloatVector.zero(F_SPECIES);
				int d       = 0;
				for (; d < hdLimit; d += F_LEN) {
					acc = FloatVector.fromArray(F_SPECIES, q,      qBase   + d)
					                 .fma(FloatVector.fromArray(F_SPECIES, kCache, kOffset + d), acc);
				}
				float dot = acc.reduceLanes(VectorOperators.ADD);
				for (; d < Hd; d++) dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}

			softmax(scores, seqLen);

			// ── V weighted sum — vectorized over headDim ──────────────────────
			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int   vOffset = t * cfg.kvDim() + kBase;
				var   vw      = FloatVector.broadcast(F_SPECIES, scores[t]);
				int   d       = 0;
				for (; d < hdLimit; d += F_LEN) {
					// out += score * vCache  ↔  fma(vw, vCache, out)
					FloatVector.fromArray(F_SPECIES, vCache, vOffset  + d)
					           .fma(vw, FloatVector.fromArray(F_SPECIES, out, outBase + d))
					           .intoArray(out, outBase + d);
				}
				float w = scores[t];
				for (; d < Hd; d++) out[outBase + d] += w * vCache[vOffset + d];
			}
		}
		return out;
	}

	/**
	 * In-place softmax over scores[0..n).
	 *
	 * <p>Vectorized passes:
	 * <ol>
	 *   <li>Max-find via {@code reduceLanes(MAX)} over vector chunks
	 *   <li>exp() — scalar (no vector transcendental)
	 *   <li>Normalize via reciprocal multiply (avoids repeated division)
	 * </ol>
	 */
	static void softmax(float[] scores, int n) {
		int limit = F_SPECIES.loopBound(n);

		// Vectorized max-find
		var maxVec = FloatVector.broadcast(F_SPECIES, Float.NEGATIVE_INFINITY);
		int i = 0;
		for (; i < limit; i += F_LEN) {
			maxVec = maxVec.max(FloatVector.fromArray(F_SPECIES, scores, i));
		}
		float max = maxVec.reduceLanes(VectorOperators.MAX);
		for (; i < n; i++) max = Math.max(max, scores[i]);

		// Scalar exp — compute sum in the same pass to avoid a third loop
		float sum = 0f;
		for (i = 0; i < n; i++) {
			scores[i] = (float) Math.exp(scores[i] - max);
			sum += scores[i];
		}

		// Vectorized normalize: multiply by 1/sum (faster than repeated division)
		var vinv = FloatVector.broadcast(F_SPECIES, 1f / sum);
		i = 0;
		for (; i < limit; i += F_LEN) {
			FloatVector.fromArray(F_SPECIES, scores, i).mul(vinv).intoArray(scores, i);
		}
		for (; i < n; i++) scores[i] /= sum;
	}

	/** SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)) */
	static float silu(float x) {
		return x / (1f + (float) Math.exp(-x));
	}

	/**
	 * Element-wise vector addition (returns new array).
	 * Vectorized with a single vmovaps + vaddps per chunk.
	 */
	static float[] add(float[] a, float[] b) {
		int     n     = a.length;
		float[] out   = new float[n];
		int     limit = F_SPECIES.loopBound(n);
		int     i     = 0;
		for (; i < limit; i += F_LEN) {
			FloatVector.fromArray(F_SPECIES, a, i)
			           .add(FloatVector.fromArray(F_SPECIES, b, i))
			           .intoArray(out, i);
		}
		for (; i < n; i++) out[i] = a[i] + b[i];
		return out;
	}

	// ── Quantized matVec overloads ────────────────────────────────────────────

	/**
	 * Matrix–vector multiply against a raw quantised weight tensor.
	 * Dispatches to the appropriate dequant+multiply implementation by GGML type.
	 */
	static float[] matVec(GgufReader.QuantizedTensor A, float[] x, int rowStart, int rowEnd, int cols) {
		MatVecEvent evt = new MatVecEvent();
		evt.begin();
		try {
			float[] y   = matVecQuantizedNoEvent(A, x, rowStart, rowEnd, cols);
			evt.backend = matVecQuantBackendLabel(A.type());
			evt.rows    = rowEnd - rowStart;
			evt.cols    = cols;
			return y;
		} finally {
			evt.commit();
		}
	}

	/**
	 * Returns the JFR backend label for a quantized matVec call.
	 *
	 * <p>All quantized overloads (matVecQ4Kraw, matVecQ6Kraw, etc.)
	 * are pure-Java CPU computations executed on
	 * ForkJoinPool.commonPool(). They never touch a GPU, so they are labelled
	 * "cpu" — the same label used by {@link CpuMatVec#sgemv}. This ensures
	 * juno.MatVec.backend.cpu.count reflects the true number of CPU-side
	 * matrix multiplies, including quantized ones.
	 *
	 * @param ggmlType GGML type ID (unused — kept for signature clarity)
	 */
	@SuppressWarnings("unused")
	private static String matVecQuantBackendLabel(int ggmlType) {
		return "cpu";
	}

	private static float[] matVecQuantizedNoEvent(GgufReader.QuantizedTensor A, float[] x,
			int rowStart, int rowEnd, int cols) {
		return switch (A.type()) {
			case 0  -> matVecF32raw(A.data(),  x, rowStart, rowEnd, cols);
			case 8  -> matVecQ8_0raw(A.data(), x, rowStart, rowEnd, cols);
			case 12 -> matVecQ4Kraw(A.data(),  x, rowStart, rowEnd, cols);
			case 13 -> matVecQ5Kraw(A.data(),  x, rowStart, rowEnd, cols);
			case 14 -> matVecQ6Kraw(A.data(),  x, rowStart, rowEnd, cols);
			default -> throw new UnsupportedOperationException(
					"Quantized matVec not implemented for GGML type " + A.type()
					+ " — add a case branch or convert to float[] first.");
		};
	}

	static float[] matVec(GgufReader.QuantizedTensor A, float[] x, int rows, int cols) {
		return matVec(A, x, 0, rows, cols);
	}

	// ── F32 raw bytes matVec ──────────────────────────────────────────────────

	/**
	 * F32 raw-byte matrix–vector multiply.
	 *
	 * <p>Original used {@code ByteBuffer.wrap(raw).getFloat(offset)} inside the
	 * parallel lambda — allocating a new ByteBuffer wrapper per row (~2048
	 * allocations per matVec call, flooding the young-gen GC).
	 *
	 * <p>Replacement uses {@link FloatVector#fromByteArray} which reads floats
	 * directly from a {@code byte[]} in a specified byte order with no object
	 * allocation, and maps to native LE load instructions on x86 (no byte-swap).
	 * The scalar tail uses explicit bit manipulation for the same reason.
	 */
	private static float[] matVecF32raw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
	    int rows  = rowEnd - rowStart;
	    float[] y = new float[rows];
	    int limit = F_SPECIES.loopBound(cols);

	    java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
	        float[] vs = F32_LOAD_SCRATCH.get(); // float[F_LEN], reused across calls
	        int base   = (rowStart + r) * cols;
	        var acc    = FloatVector.zero(F_SPECIES);
	        int c      = 0;
	        for (; c < limit; c += F_LEN) {
	            // Fill one vector's worth of floats from LE bytes — F_LEN iterations only
	            int byteBase = (base + c) * 4;
	            for (int j = 0; j < F_LEN; j++) {
	                int off  = byteBase + j * 4;
	                int bits = (raw[off]     & 0xFF)
	                         | ((raw[off+1] & 0xFF) << 8)
	                         | ((raw[off+2] & 0xFF) << 16)
	                         | ((raw[off+3] & 0xFF) << 24);
	                vs[j] = Float.intBitsToFloat(bits);
	            }
	            acc = FloatVector.fromArray(F_SPECIES, vs, 0)
	                             .fma(FloatVector.fromArray(F_SPECIES, x, c), acc);
	        }
	        float sum = acc.reduceLanes(VectorOperators.ADD);
	        for (; c < cols; c++) {
	            int off  = (base + c) * 4;
	            int bits = (raw[off]     & 0xFF)
	                     | ((raw[off+1] & 0xFF) << 8)
	                     | ((raw[off+2] & 0xFF) << 16)
	                     | ((raw[off+3] & 0xFF) << 24);
	            sum += Float.intBitsToFloat(bits) * x[c];
	        }
	        y[r] = sum;
	    });
	    return y;
	}

	// ── Q8_0 raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q8_0 block-wise matrix–vector multiply.
	 *
	 * <p>Block layout: [scale:f16(2)][qs:32 signed bytes] = 34 bytes / 32 elements.
	 *
	 * <p>Vectorization: within each 32-element block, the signed bytes are
	 * dequantized into a thread-local float[32] scratch, then accumulated via
	 * vector FMA. The scratch eliminates per-block heap allocation inside the
	 * already-parallel row lambda.
	 */
	private static float[] matVecQ8_0raw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE   = 32;
		final int BLOCK_BYTES  = 34;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		int   rows = rowEnd - rowStart;
		float[] y  = new float[rows];
		int limit  = F_SPECIES.loopBound(BLOCK_SIZE); // loopBound(32): 32 for AVX2, 32 for AVX-512

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			float[] ws           = Q8_SCRATCH.get(); // thread-local float[32], never null
			int rowByteOffset    = (rowStart + r) * bytesPerRow;
			float acc            = 0f;
			int xBase            = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int   bo    = rowByteOffset + b * BLOCK_BYTES;
				float scale = GgufReader.f16ToF32(readLE16(raw, bo));

				// Dequantize 32 signed bytes → float[] via scale
				for (int i = 0; i < BLOCK_SIZE; i++) ws[i] = raw[bo + 2 + i] * scale;

				// Vector FMA: acc += ws[i] * x[xBase + i]
				var vacc = FloatVector.zero(F_SPECIES);
				int i    = 0;
				for (; i < limit; i += F_LEN) {
					vacc = FloatVector.fromArray(F_SPECIES, ws,    i)
					                  .fma(FloatVector.fromArray(F_SPECIES, x, xBase + i), vacc);
				}
				acc += vacc.reduceLanes(VectorOperators.ADD);
				for (; i < BLOCK_SIZE; i++) acc += ws[i] * x[xBase + i];

				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	/** Read a little-endian signed 16-bit value from raw bytes at offset. */
	static short readLE16(byte[] raw, int offset) {
		return (short) ((raw[offset] & 0xFF) | ((raw[offset + 1] & 0xFF) << 8));
	}

	// ── Q4_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q4_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [d:f16(2)][dmin:f16(2)][sc:12][qs:128] = 144 bytes per 256
	 * elements. 4 groups of 64 elements; each group yields two 32-element
	 * sub-blocks via low/high nibbles — matching llama.cpp dequantize_row_q4_K.
	 *
	 * The dequant inner loops operate on 32 elements at a time; nibble extraction
	 * is not SIMD-friendly, so rows are parallelized across ForkJoin workers and
	 * the JIT auto-vectorizes where possible.
	 */
	private static float[] matVecQ4Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE   = 256;
		final int BLOCK_BYTES  = 144;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		int   rows = rowEnd - rowStart;
		float[] y  = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo     = rowByteOffset + b * BLOCK_BYTES;
				int scBase = bo + 4;   // 12 scale bytes at [bo+4, bo+15]
				int qsBase = bo + 16;  // 128 nibble bytes at [bo+16, bo+143]
				float d    = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));

				int qi = 0;
				for (int g = 0; g < BLOCK_SIZE; g += 64) {
					int   s0     = g / 32;
					int   s1     = s0 + 1;
					float scale0 = d    * q4kScaleRaw(raw, scBase, s0);
					float min0   = dmin * q4kMinRaw(raw, scBase, s0);
					float scale1 = d    * q4kScaleRaw(raw, scBase, s1);
					float min1   = dmin * q4kMinRaw(raw, scBase, s1);

					for (int i = 0; i < 32; i++)
						acc += (scale0 * (raw[qsBase + qi + i] & 0x0F) - min0) * x[xBase + g + i];
					for (int i = 0; i < 32; i++)
						acc += (scale1 * ((raw[qsBase + qi + i] >> 4) & 0x0F) - min1) * x[xBase + g + 32 + i];
					qi += 32;
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	/** Q4_K scale[j]: reads 6-bit packed value directly from raw[] at scBase. */
	static float q4kScaleRaw(byte[] raw, int scBase, int j) {
		int v = (j < 4) ? raw[scBase + j] & 0x3F
				: ((raw[scBase + j + 4] & 0x0F) | ((raw[scBase + j - 4] & 0xC0) >> 2)) & 0x3F;
		return v;
	}

	/** Q4_K min[j]: reads 6-bit packed value directly from raw[] at scBase. */
	static float q4kMinRaw(byte[] raw, int scBase, int j) {
		int v = (j < 4) ? raw[scBase + j + 4] & 0x3F
				: (((raw[scBase + j + 4] & 0xFF) >> 4) | ((raw[scBase + j] & 0xC0) >> 2)) & 0x3F;
		return v;
	}

	// ── Q5_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q5_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [d:f16(2)][dmin:f16(2)][sc:12][qh:32][qs:128] = 176 bytes per
	 * 256 elements. 5th bit per element stored in qh: value = (nibble | (qh_bit << 4)).
	 */
	private static float[] matVecQ5Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE   = 256;
		final int BLOCK_BYTES  = 176;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		int   rows = rowEnd - rowStart;
		float[] y  = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo     = rowByteOffset + b * BLOCK_BYTES;
				int scBase = bo + 4;   // 12 scale bytes [bo+4, bo+15]
				int qhBase = bo + 16;  // 32 hi-bit bytes [bo+16, bo+47]
				int qsBase = bo + 48;  // 128 nibble bytes[bo+48, bo+175]
				float d    = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));

				int qi = 0;
				for (int g = 0; g < 4; g++) {
					int   s0     = g * 2;
					int   s1     = s0 + 1;
					float scale0 = d    * q4kScaleRaw(raw, scBase, s0);
					float min0   = dmin * q4kMinRaw(raw, scBase, s0);
					float scale1 = d    * q4kScaleRaw(raw, scBase, s1);
					float min1   = dmin * q4kMinRaw(raw, scBase, s1);
					int hiBit0   = g * 2;
					int hiBit1   = g * 2 + 1;

					for (int l = 0; l < 32; l++) {
						int lo = raw[qsBase + qi + l] & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit0) & 1;
						acc += (scale0 * (lo | (hi << 4)) - min0) * x[xBase + g * 64 + l];
					}
					for (int l = 0; l < 32; l++) {
						int lo = (raw[qsBase + qi + l] >>> 4) & 0x0F;
						int hi = (raw[qhBase + l] >>> hiBit1) & 1;
						acc += (scale1 * (lo | (hi << 4)) - min1) * x[xBase + g * 64 + 32 + l];
					}
					qi += 32;
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	// ── Q6_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q6_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [ql:128][qh:64][sc:16][d:f16] = 210 bytes per 256 elements.
	 * Signed 6-bit values in [-32,31], scaled by d * sc[].
	 */
	private static float[] matVecQ6Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE   = 256;
		final int BLOCK_BYTES  = 210;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow  = blocksPerRow * BLOCK_BYTES;
		int   rows = rowEnd - rowStart;
		float[] y  = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				// layout: ql[128] at bo, qh[64] at bo+128, sc[16] at bo+192, d:f16 at bo+208
				float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));

				for (int half = 0; half < 2; half++) {
					int qlOff = bo + half * 64;
					int qhOff = bo + 128 + half * 32;
					int scOff = bo + 192 + half * 8;
					int xOff  = xBase + half * 128;

					for (int l = 0; l < 32; l++) {
						int is   = l / 16;
						int qlL  = raw[qlOff + l] & 0xFF;
						int qlL2 = raw[qlOff + l + 32] & 0xFF;
						int qhL  = raw[qhOff + l] & 0xFF;

						int q1 = ((qlL & 0x0F)  | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL >> 4)    | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4)   | (((qhL >> 6) & 3) << 4)) - 32;

						float d1 = d * raw[scOff + is];
						float d2 = d * raw[scOff + is + 2];
						float d3 = d * raw[scOff + is + 4];
						float d4 = d * raw[scOff + is + 6];

						acc += d1 * q1 * x[xOff + l];
						acc += d2 * q2 * x[xOff + l + 32];
						acc += d3 * q3 * x[xOff + l + 64];
						acc += d4 * q4 * x[xOff + l + 96];
					}
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}
}