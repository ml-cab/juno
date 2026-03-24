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
 * WITHOUT WARRANTIES OR ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * GPU-accelerated LLaMA-family transformer forward pass.
 *
 * Structurally mirrors CpuForwardPassHandler. The only difference is that
 * all matrix-vector multiplies (matVec) are delegated to a GpuMatVec
 * instance — in production a CublasMatVec backed by cublasSgemv, in tests
 * a CpuMatVec or a custom stub.
 *
 * Weight loading, layer loop, attention, FFN, RoPE, RMS norm and KV cache
 * management are identical to the CPU version. This ensures that GPU and CPU
 * nodes produce numerically equivalent results (within float32 rounding).
 *
 * Thread safety: each request uses an isolated KV cache keyed by requestId.
 * Multiple threads may call forward() concurrently for distinct requestIds.
 *
 * Typical wiring on a GPU node:
 *   GpuContext ctx = GpuContext.init(0);
 *   GpuForwardPassHandler handler = GpuForwardPassHandler.loadGpuResident(modelPath, shard, ctx);
 *
 * {@link #load(Path, ShardContext, GpuMatVec)} keeps matmul weights on the host
 * (used with CpuMatVec or for tests). {@link #loadGpuResident} uploads each
 * weight matrix once to the device for production inference.
 */
public final class GpuForwardPassHandler implements ForwardPassHandler {

    private static final Logger log =
        Logger.getLogger(GpuForwardPassHandler.class.getName());

    // ── Loaded weights ────────────────────────────────────────────────────────

    private final LlamaConfig cfg;
    private final int startLayer;
    private final int endLayer;
    private final boolean hasEmbeddings;
    private final boolean hasOutputProj;

    /** True when matmul weights live on the GPU ({@link DeviceFloatMatrix}). */
    private final boolean residentGpuWeights;

    private final float[] tokenEmbd;
    private final float[] outputNorm;
    /** Host output projection; null when {@link #residentGpuWeights} (use {@link #outputProjD}). */
    private final float[] outputProj;

    private final float[][] attnNorm;
    private final float[][] ffnNorm;

    /** Host matmul weights; null when {@link #residentGpuWeights}. */
    private final float[][] wq;
    private final float[][] wk;
    private final float[][] wv;
    private final float[][] wo;
    private final float[][] wGate;
    private final float[][] wUp;
    private final float[][] wDown;

    /** Device matmul weights; null when not {@link #residentGpuWeights}. */
    private final DeviceFloatMatrix[] wqD;
    private final DeviceFloatMatrix[] wkD;
    private final DeviceFloatMatrix[] wvD;
    private final DeviceFloatMatrix[] woD;
    private final DeviceFloatMatrix[] wGateD;
    private final DeviceFloatMatrix[] wUpD;
    private final DeviceFloatMatrix[] wDownD;

    private final DeviceFloatMatrix outputProjD;

    // Per-request KV cache — same layout as CpuForwardPassHandler
    private final Map<String, float[][]> kvCacheK = new HashMap<>();
    private final Map<String, float[][]> kvCacheV = new HashMap<>();
    private static final int MAX_SEQ_LEN = 2048;

    private final GpuMatVec matVec;

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Load weights from a GGUF file and wire up the given GpuMatVec backend.
     *
     * <p>Weights stay in host memory; each matmul uploads the matrix to the GPU
     * when using {@link CublasMatVec}. Prefer {@link #loadGpuResident} for
     * production GPU nodes.
     *
     * @param modelPath path to the GGUF model file
     * @param context   shard assignment for this node
     * @param matVec    the matmul backend — CublasMatVec on GPU, CpuMatVec as fallback
     */
    public static GpuForwardPassHandler load(
            Path modelPath, ShardContext context, GpuMatVec matVec) throws IOException {

        log.info("GpuForwardPassHandler loading: layers "
            + context.startLayer() + "–" + context.endLayer()
            + "  embd=" + context.hasEmbeddings()
            + "  outProj=" + context.hasOutputProjection()
            + "  file=" + modelPath
            + "  backend=" + matVec.getClass().getSimpleName()
            + "  hostWeights=true");

        try (GgufReader r = GgufReader.open(modelPath)) {
            LlamaConfig cfg = LlamaConfig.from(r);
            return new GpuForwardPassHandler(r, cfg, context, matVec);
        }
    }

    /**
     * Load weights and upload all matmul tensors to the GPU once.
     *
     * <p>Requires CUDA. Uses {@link CublasMatVec} internally. Call
     * {@link #releaseGpuResources()} before destroying {@link GpuContext}.
     *
     * @param modelPath path to the GGUF model file
     * @param context   shard assignment for this node
     * @param gpuCtx    CUDA / cuBLAS context for this JVM
     */
    public static GpuForwardPassHandler loadGpuResident(
            Path modelPath, ShardContext context, GpuContext gpuCtx) throws IOException {

        if (gpuCtx == null)
            throw new IllegalArgumentException("gpuCtx must not be null");
        if (!CudaAvailability.isAvailable())
            throw new IllegalStateException("CUDA not available — cannot loadGpuResident");

        log.info("GpuForwardPassHandler loadGpuResident: layers "
            + context.startLayer() + "–" + context.endLayer()
            + "  embd=" + context.hasEmbeddings()
            + "  outProj=" + context.hasOutputProjection()
            + "  file=" + modelPath
            + "  hostWeights=false");

        CublasMatVec matVec = new CublasMatVec(gpuCtx);
        try (GgufReader r = GgufReader.open(modelPath)) {
            LlamaConfig cfg = LlamaConfig.from(r);
            return new GpuForwardPassHandler(r, cfg, context, matVec, gpuCtx);
        }
    }

    private GpuForwardPassHandler(
            GgufReader r, LlamaConfig cfg, ShardContext ctx, GpuMatVec matVec)
            throws IOException {

        this.cfg = cfg;
        this.startLayer = ctx.startLayer();
        this.endLayer = ctx.endLayer();
        this.hasEmbeddings = ctx.hasEmbeddings();
        this.hasOutputProj = ctx.hasOutputProjection();
        this.matVec = matVec;
        this.residentGpuWeights = false;

        int L = endLayer - startLayer;

        this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
        this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
        this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

        this.outputProjD = null;
        this.wqD = this.wkD = this.wvD = this.woD = null;
        this.wGateD = this.wUpD = this.wDownD = null;

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

    private GpuForwardPassHandler(
            GgufReader r, LlamaConfig cfg, ShardContext ctx, CublasMatVec matVec, GpuContext gpuCtx)
            throws IOException {

        this.cfg = cfg;
        this.startLayer = ctx.startLayer();
        this.endLayer = ctx.endLayer();
        this.hasEmbeddings = ctx.hasEmbeddings();
        this.hasOutputProj = ctx.hasOutputProjection();
        this.matVec = matVec;
        this.residentGpuWeights = true;

        int L = endLayer - startLayer;
        int H = cfg.hiddenDim();
        int kvDim = cfg.kvDim();
        int I = cfg.intermediateSize();

        this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
        this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
        this.outputProj = null;

        this.wq = this.wk = this.wv = this.wo = null;
        this.wGate = this.wUp = this.wDown = null;

        attnNorm = new float[L][];
        ffnNorm = new float[L][];
        wqD = new DeviceFloatMatrix[L];
        wkD = new DeviceFloatMatrix[L];
        wvD = new DeviceFloatMatrix[L];
        woD = new DeviceFloatMatrix[L];
        wGateD = new DeviceFloatMatrix[L];
        wUpD = new DeviceFloatMatrix[L];
        wDownD = new DeviceFloatMatrix[L];

        List<DeviceFloatMatrix> uploaded = new ArrayList<>();

        try {
            for (int li = 0; li < L; li++) {
                int i = li + startLayer;
                attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
                ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");

                wqD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".attn_q.weight"), H, H);
                uploaded.add(wqD[li]);
                wkD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".attn_k.weight"), kvDim, H);
                uploaded.add(wkD[li]);
                wvD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".attn_v.weight"), kvDim, H);
                uploaded.add(wvD[li]);
                woD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".attn_output.weight"), H, H);
                uploaded.add(woD[li]);
                wGateD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".ffn_gate.weight"), I, H);
                uploaded.add(wGateD[li]);
                wUpD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".ffn_up.weight"), I, H);
                uploaded.add(wUpD[li]);
                wDownD[li] = DeviceFloatMatrix.upload(gpuCtx, r.tensor("blk." + i + ".ffn_down.weight"), H, I);
                uploaded.add(wDownD[li]);
            }

            if (hasOutputProj) {
                float[] opHost = loadOutputProjection(r);
                outputProjD = DeviceFloatMatrix.upload(gpuCtx, opHost, cfg.vocabSize(), H);
                uploaded.add(outputProjD);
            } else {
                outputProjD = null;
            }

            log.info("GpuForwardPassHandler loaded (GPU-resident weights) — " + L + " layers");
        } catch (Exception e) {
            for (DeviceFloatMatrix d : uploaded) {
                try {
                    d.close();
                } catch (Exception ignored) {
                    /* best effort */
                }
            }
            if (e instanceof IOException ioe) throw ioe;
            if (e instanceof RuntimeException re) throw re;
            throw new IOException(e);
        }
    }

    /**
     * Frees all {@link DeviceFloatMatrix} buffers created by {@link #loadGpuResident}.
     * No-op for handlers loaded with {@link #load}. Safe to call multiple times.
     */
    public void releaseGpuResources() {
        if (!residentGpuWeights) return;
        closeAll(wqD);
        closeAll(wkD);
        closeAll(wvD);
        closeAll(woD);
        closeAll(wGateD);
        closeAll(wUpD);
        closeAll(wDownD);
        if (outputProjD != null) outputProjD.close();
    }

    private static void closeAll(DeviceFloatMatrix[] arr) {
        if (arr == null) return;
        for (DeviceFloatMatrix d : arr) {
            if (d != null) d.close();
        }
    }

    private static float[] loadOutputProjection(GgufReader r) throws IOException {
        if (r.hasTensor("output.weight")) return r.tensor("output.weight");
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

    private float[] transformerLayer(
            float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {

        int H = cfg.hiddenDim();

        float[] xNorm = CpuForwardPassHandler.rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

        float[] q = residentGpuWeights
            ? matVec.sgemv(wqD[li], xNorm)
            : matVec.sgemv(wq[li], xNorm, H, H);
        float[] k = residentGpuWeights
            ? matVec.sgemv(wkD[li], xNorm)
            : matVec.sgemv(wk[li], xNorm, cfg.kvDim(), H);
        float[] v = residentGpuWeights
            ? matVec.sgemv(wvD[li], xNorm)
            : matVec.sgemv(wv[li], xNorm, cfg.kvDim(), H);

        CpuForwardPassHandler.rope(q, pos, cfg.numHeads(), cfg.headDim(), cfg.ropeTheta());
        CpuForwardPassHandler.rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

        System.arraycopy(k, 0, kCacheLayer, pos * cfg.kvDim(), cfg.kvDim());
        System.arraycopy(v, 0, vCacheLayer, pos * cfg.kvDim(), cfg.kvDim());

        float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
        float[] attnProj = residentGpuWeights
            ? matVec.sgemv(woD[li], attnOut)
            : matVec.sgemv(wo[li], attnOut, H, H);
        float[] x2 = CpuForwardPassHandler.add(x, attnProj);

        float[] xNorm2 = CpuForwardPassHandler.rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
        float[] ffnOut = ffn(xNorm2, li);
        return CpuForwardPassHandler.add(x2, ffnOut);
    }

    private float[] ffn(float[] x, int li) {
        int H = cfg.hiddenDim();
        int I = cfg.intermediateSize();
        float[] gate = residentGpuWeights
            ? matVec.sgemv(wGateD[li], x)
            : matVec.sgemv(wGate[li], x, I, H);
        float[] up = residentGpuWeights
            ? matVec.sgemv(wUpD[li], x)
            : matVec.sgemv(wUp[li], x, I, H);
        float[] hidden = new float[I];
        for (int i = 0; i < I; i++)
            hidden[i] = CpuForwardPassHandler.silu(gate[i]) * up[i];
        return residentGpuWeights
            ? matVec.sgemv(wDownD[li], hidden)
            : matVec.sgemv(wDown[li], hidden, H, I);
    }

    private float[] outputProjection(float[] x) {
        float[] xNorm = CpuForwardPassHandler.rmsNorm(x, outputNorm, cfg.rmsNormEps());
        return residentGpuWeights
            ? matVec.sgemv(outputProjD, xNorm)
            : matVec.sgemv(outputProj, xNorm, cfg.vocabSize(), cfg.hiddenDim());
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
