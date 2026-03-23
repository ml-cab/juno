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
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * GPU-accelerated LLaMA-family transformer forward pass.
 *
 * Structurally mirrors CpuForwardPassHandler. The only difference is that all
 * matrix-vector multiplies (matVec) are delegated to a GpuMatVec instance — in
 * production a CublasMatVec backed by cublasSgemv, in tests a CpuMatVec or a
 * custom stub.
 *
 * Weight loading, layer loop, attention, FFN, RoPE, RMS norm and KV cache
 * management are identical to the CPU version. This ensures that GPU and CPU
 * nodes produce numerically equivalent results (within float32 rounding).
 *
 * Thread safety: each request uses an isolated KV cache keyed by requestId.
 * Multiple threads may call forward() concurrently for distinct requestIds.
 *
 * Typical wiring on a GPU node: GpuContext ctx = GpuContext.init(0); GpuMatVec
 * matVec = new CublasMatVec(ctx); GpuForwardPassHandler handler =
 * GpuForwardPassHandler.load(modelPath, shard, matVec);
 */
public final class GpuForwardPassHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(GpuForwardPassHandler.class.getName());

	// ── Loaded weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	private final float[] tokenEmbd;
	private final float[] outputNorm;
	private final float[] outputProj;

	private final float[][] attnNorm;
	private final float[][] ffnNorm;
	private final float[][] wq;
	private final float[][] wk;
	private final float[][] wv;
	private final float[][] wo;
	private final float[][] wGate;
	private final float[][] wUp;
	private final float[][] wDown;

	// Per-request KV cache — same layout as CpuForwardPassHandler
	private final Map<String, float[][]> kvCacheK = new HashMap<>();
	private final Map<String, float[][]> kvCacheV = new HashMap<>();
	private static final int MAX_SEQ_LEN = 2048;

	// ── The only difference from CpuForwardPassHandler ────────────────────────
	private final GpuMatVec matVec;

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load weights from a GGUF file and wire up the given GpuMatVec backend.
	 *
	 * @param modelPath path to the GGUF model file
	 * @param context   shard assignment for this node
	 * @param matVec    the matmul backend — CublasMatVec on GPU, CpuMatVec as
	 *                  fallback
	 */
	public static GpuForwardPassHandler load(Path modelPath, ShardContext context, GpuMatVec matVec)
			throws IOException {

		log.info("GpuForwardPassHandler loading: layers " + context.startLayer() + "–" + context.endLayer() + "  embd="
				+ context.hasEmbeddings() + "  outProj=" + context.hasOutputProjection() + "  file=" + modelPath
				+ "  backend=" + matVec.getClass().getSimpleName());

		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new GpuForwardPassHandler(r, cfg, context, matVec);
		}
	}

	private GpuForwardPassHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, GpuMatVec matVec)
			throws IOException {

		this.cfg = cfg;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();
		this.matVec = matVec;

		int L = endLayer - startLayer;

		this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

		attnNorm = new float[L][];
		ffnNorm = new float[L][];
		wq = new float[L][];
		wk = new float[L][];
		wv = new float[L][];
		wo = new float[L][];
		wGate = new float[L][];
		wUp = new float[L][];
		wDown = new float[L][];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");
			wq[li] = r.tensor("blk." + i + ".attn_q.weight");
			wk[li] = r.tensor("blk." + i + ".attn_k.weight");
			wv[li] = r.tensor("blk." + i + ".attn_v.weight");
			wo[li] = r.tensor("blk." + i + ".attn_output.weight");
			wGate[li] = r.tensor("blk." + i + ".ffn_gate.weight");
			wUp[li] = r.tensor("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensor("blk." + i + ".ffn_down.weight");
		}

		log.info("GpuForwardPassHandler loaded — " + L + " layers");
	}

	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight"))
			return r.tensor("output.weight");
		log.info("output.weight absent — using tied embeddings");
		return r.tensor("token_embd.weight");
	}

	// ── ForwardPassHandler ────────────────────────────────────────────────────

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long start = System.nanoTime();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			return ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
		}
	}

	@Override
	public boolean isReady() {
		return true;
	}

	// ── Transformer forward pass — identical logic to CpuForwardPassHandler ──

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
		int L = endLayer - startLayer;
		kvCacheK.computeIfAbsent(requestId, _ -> new float[L][MAX_SEQ_LEN * cfg.kvDim()]);
		kvCacheV.computeIfAbsent(requestId, _ -> new float[L][MAX_SEQ_LEN * cfg.kvDim()]);

		float[][] kCache = kvCacheK.get(requestId);
		float[][] vCache = kvCacheV.get(requestId);

		for (int li = 0; li < L; li++) {
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);
		}
		return x;
	}

	private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {

		int H = cfg.hiddenDim();

		// Attention
		float[] xNorm = CpuForwardPassHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		// ── matVec calls go to GPU via GpuMatVec ──────────────────────────────
		float[] q = matVec.sgemv(wq[li], xNorm, H, H);
		float[] k = matVec.sgemv(wk[li], xNorm, cfg.kvDim(), H);
		float[] v = matVec.sgemv(wv[li], xNorm, cfg.kvDim(), H);

		CpuForwardPassHandler.rope(q, pos, cfg.numHeads(), cfg.headDim(), cfg.ropeTheta());
		CpuForwardPassHandler.rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		System.arraycopy(k, 0, kCacheLayer, pos * cfg.kvDim(), cfg.kvDim());
		System.arraycopy(v, 0, vCacheLayer, pos * cfg.kvDim(), cfg.kvDim());

		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
		float[] attnProj = matVec.sgemv(wo[li], attnOut, H, H); // ── GPU
		float[] x2 = CpuForwardPassHandler.add(x, attnProj);

		// FFN
		float[] xNorm2 = CpuForwardPassHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return CpuForwardPassHandler.add(x2, ffnOut);
	}

	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = matVec.sgemv(wGate[li], x, I, H); // ── GPU
		float[] up = matVec.sgemv(wUp[li], x, I, H); // ── GPU
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = CpuForwardPassHandler.silu(gate[i]) * up[i];
		return matVec.sgemv(wDown[li], hidden, H, I); // ── GPU
	}

	private float[] outputProjection(float[] x) {
		float[] xNorm = CpuForwardPassHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
		return matVec.sgemv(outputProj, xNorm, cfg.vocabSize(), cfg.hiddenDim()); // ── GPU
	}

	// ── GQA — pure Java, identical to CpuForwardPassHandler ─────────────────

	private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
		int H = cfg.numHeads();
		@SuppressWarnings("unused")
		int KVH = cfg.numKvHeads();
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

			CpuForwardPassHandler.softmax(scores, seqLen);

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