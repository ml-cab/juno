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
 * Phi-2 family transformer forward pass (moondream2, phi-2, etc.).
 *
 * <h3>Phi-2 vs LLaMA / Phi-3 tensor layout differences</h3>
 * <ol>
 * <li><b>LayerNorm (not RMSNorm)</b>: Phi-2 uses standard layer normalisation
 *     with both weight <em>and</em> bias:
 *     {@code y = (x − mean) / sqrt(var + ε) * weight + bias}.
 *     The tensor name is {@code blk.{i}.attn_norm.weight} / {@code .attn_norm.bias}.
 *     There is <em>no</em> separate {@code ffn_norm} tensor.
 * <li><b>Parallel attention + FFN</b>: both sub-layers share the <em>same</em>
 *     {@code layerNorm(x)} output and their results are accumulated into a
 *     <em>single</em> residual addition:
 *     {@code x = x + attn_out + ffn_out}.
 *     This is why {@code ffn_norm.weight} is absent — it would be redundant.
 * <li><b>Fused QKV with bias</b>: {@code blk.{i}.attn_qkv.weight}
 *     shape {@code [H + 2·kvDim, H]} plus {@code attn_qkv.bias [H + 2·kvDim]}.
 * <li><b>Attention output bias</b>: {@code blk.{i}.attn_output.bias [H]}.
 * <li><b>GELU FFN (no gate)</b>: only {@code ffn_up} and {@code ffn_down}
 *     tensors; no {@code ffn_gate}. Activation is GELU, not SiLU.
 *     Both projections carry bias vectors.
 * <li><b>Partial RoPE</b>: rotates only the first {@code phi2.rope.dimension_count}
 *     (typically 32) dimensions of each head; the remaining dims are unchanged.
 * <li><b>Output projection bias</b>: {@code output.bias [vocabSize]}.
 * <li><b>Output LayerNorm bias</b>: {@code output_norm.bias [hiddenDim]}.
 * </ol>
 *
 * <h3>Root cause this class fixes</h3>
 * Before this class existed, the {@code phi2} architecture fell through to
 * {@link LlamaTransformerHandler} in {@link ForwardPassHandlerLoader}. That
 * handler immediately tries to load {@code blk.0.ffn_norm.weight}, which does
 * not exist in Phi-2's GGUF, producing:
 * <pre>
 *   IllegalArgumentException: Tensor not found: blk.0.ffn_norm.weight
 * </pre>
 */
public final class Phi2TransformerHandler implements ForwardPassHandler {

    private static final Logger log = Logger.getLogger(Phi2TransformerHandler.class.getName());

    // ── Config ────────────────────────────────────────────────────────────────

    private final LlamaConfig cfg;
    /** Number of head dimensions that receive RoPE (phi2.rope.dimension_count). */
    private final int ropeDim;
    private final int startLayer;
    private final int endLayer;
    private final boolean hasEmbeddings;
    private final boolean hasOutputProj;

    // ── Embedding / output weights (conditional on shard position) ────────────

    private final float[]                     tokenEmbd;      // [vocabSize * H] – first node
    private final float[]                     outputNorm;     // [H]             – last node
    private final float[]                     outputNormBias; // [H]             – last node (LayerNorm bias)
    private final GgufReader.QuantizedTensor  outputProj;     // [vocabSize * H] – last node (quantised)
    private final float[]                     outputBias;     // [vocabSize]     – last node (null if absent)

    // ── Per-layer weights ─────────────────────────────────────────────────────

    /** LayerNorm weights shared by attention and FFN (F32, one per block). */
    private final float[][] attnNorm;           // [L][H]
    private final float[][] attnNormBias;       // [L][H]

    /** Fused QKV projection: shape [H + kvDim + kvDim, H]. */
    private final GgufReader.QuantizedTensor[] attnQkv;       // [L]
    private final float[][]                    attnQkvBias;   // [L][H + kvDim + kvDim]

    /** Attention output projection: shape [H, H]. */
    private final GgufReader.QuantizedTensor[] wo;            // [L]
    private final float[][]                    woBias;        // [L][H]

    /** FFN up-projection (no gate): shape [I, H]. */
    private final GgufReader.QuantizedTensor[] wUp;           // [L]
    private final float[][]                    wUpBias;       // [L][I]

    /** FFN down-projection: shape [H, I]. */
    private final GgufReader.QuantizedTensor[] wDown;         // [L]
    private final float[][]                    wDownBias;     // [L][H]

    // ── KV cache ─────────────────────────────────────────────────────────────

    private final Map<String, float[][]> kvCacheK = new ConcurrentHashMap<>();
    private final Map<String, float[][]> kvCacheV = new ConcurrentHashMap<>();
    private static final int MAX_SEQ_LEN         = 2048;
    private static final int INITIAL_SEQ_CAPACITY = 64;   // grows on demand

    private final MatVec backend;
    private volatile NodeKVCacheAdapter kvAdapter;

    // ── Factory ───────────────────────────────────────────────────────────────

    public static Phi2TransformerHandler load(Path modelPath, ShardContext context) throws IOException {
        return load(modelPath, context, CpuMatVec.INSTANCE);
    }

    public static Phi2TransformerHandler load(Path modelPath, ShardContext context, MatVec backend)
            throws IOException {
        log.info("Loading Phi-2 GGUF shard: layers " + context.startLayer() + "–" + context.endLayer()
                + "  embd=" + context.hasEmbeddings()
                + "  outProj=" + context.hasOutputProjection()
                + "  backend=" + backend.getClass().getSimpleName()
                + "  file=" + modelPath);
        try (GgufReader r = GgufReader.open(modelPath)) {
            LlamaConfig cfg = LlamaConfig.from(r);
            log.info("Model: " + cfg);
            return new Phi2TransformerHandler(r, cfg, context, backend);
        }
    }

    private Phi2TransformerHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx, MatVec backend)
            throws IOException {
        this.cfg        = cfg;
        this.backend    = backend;
        this.startLayer = ctx.startLayer();
        this.endLayer   = ctx.endLayer();
        this.hasEmbeddings  = ctx.hasEmbeddings();
        this.hasOutputProj  = ctx.hasOutputProjection();

        // Partial RoPE: Phi-2 rotates only the first ropeDim dims of each head.
        // Default to full headDim when the metadata key is absent.
        this.ropeDim = r.metaInt("phi2.rope.dimension_count", cfg.headDim());

        int L    = endLayer - startLayer;
        int H    = cfg.hiddenDim();
        int kvDim = cfg.kvDim();
        int I    = cfg.intermediateSize();

        // ── Embedding / output weights ────────────────────────────────────────
        this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;

        if (hasOutputProj) {
            this.outputNorm     = r.tensor("output_norm.weight");
            this.outputNormBias = r.hasTensor("output_norm.bias")
                    ? r.tensor("output_norm.bias") : null;
            // Prefer output.weight; fall back to tied embeddings when absent.
            this.outputProj = r.hasTensor("output.weight")
                    ? r.tensorRaw("output.weight")
                    : r.tensorRaw("token_embd.weight");
            this.outputBias = r.hasTensor("output.bias")
                    ? r.tensor("output.bias") : null;
        } else {
            this.outputNorm     = null;
            this.outputNormBias = null;
            this.outputProj     = null;
            this.outputBias     = null;
        }

        // ── Per-layer weights ─────────────────────────────────────────────────
        attnNorm     = new float[L][];
        attnNormBias = new float[L][];
        attnQkv      = new GgufReader.QuantizedTensor[L];
        attnQkvBias  = new float[L][];
        wo           = new GgufReader.QuantizedTensor[L];
        woBias       = new float[L][];
        wUp          = new GgufReader.QuantizedTensor[L];
        wUpBias      = new float[L][];
        wDown        = new GgufReader.QuantizedTensor[L];
        wDownBias    = new float[L][];

        for (int li = 0; li < L; li++) {
            int i = li + startLayer;
            log.fine("Loading Phi-2 layer " + i);

            // Single LayerNorm shared by both attention and FFN paths.
            attnNorm[li]     = r.tensor("blk." + i + ".attn_norm.weight");
            attnNormBias[li] = r.hasTensor("blk." + i + ".attn_norm.bias")
                    ? r.tensor("blk." + i + ".attn_norm.bias") : null;

            // Fused QKV: [H + kvDim + kvDim, H]
            attnQkv[li]     = r.tensorRaw("blk." + i + ".attn_qkv.weight");
            attnQkvBias[li] = r.hasTensor("blk." + i + ".attn_qkv.bias")
                    ? r.tensor("blk." + i + ".attn_qkv.bias") : null;

            // Attention output: [H, H]
            wo[li]     = r.tensorRaw("blk." + i + ".attn_output.weight");
            woBias[li] = r.hasTensor("blk." + i + ".attn_output.bias")
                    ? r.tensor("blk." + i + ".attn_output.bias") : null;

            // FFN: up [I, H], down [H, I] — no gate tensor in Phi-2.
            wUp[li]     = r.tensorRaw("blk." + i + ".ffn_up.weight");
            wUpBias[li] = r.hasTensor("blk." + i + ".ffn_up.bias")
                    ? r.tensor("blk." + i + ".ffn_up.bias") : null;

            wDown[li]     = r.tensorRaw("blk." + i + ".ffn_down.weight");
            wDownBias[li] = r.hasTensor("blk." + i + ".ffn_down.bias")
                    ? r.tensor("blk." + i + ".ffn_down.bias") : null;
        }

        log.info("Phi-2 shard loaded — " + L + " layers"
                + (hasEmbeddings  ? ", with embeddings"        : "")
                + (hasOutputProj  ? ", with output projection" : "")
                + "  ropeDim=" + ropeDim + "/" + cfg.headDim());
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
            result = ForwardResult.logits(request.requestId(), outputProjection(x), System.nanoTime() - start);
        } else {
            result = ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
        }

        evt.handlerType         = "phi2";
        evt.requestId           = request.requestId();
        evt.startPosition       = request.startPosition();
        evt.layerCount          = endLayer - startLayer;
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
        // Return the post-norm hidden state (LayerNorm, consistent with Phi-2 training).
        return Optional.of(layerNorm(x, outputNorm, outputNormBias, cfg.rmsNormEps()));
    }

    @Override
    public boolean isReady() {
        return true;
    }

    @Override
    public float[] embedToken(int tokenId) {
        if (!hasEmbeddings)
            throw new UnsupportedOperationException(
                    "embedToken() called on a shard with hasEmbeddings=false");
        int H = cfg.hiddenDim();
        int clamped = Math.max(0, Math.min(tokenId, cfg.vocabSize() - 1));
        float[] x = new float[H];
        System.arraycopy(tokenEmbd, clamped * H, x, 0, H);
        return x;
    }

    // Phi-2 is CPU-only in this implementation; no GPU resources to release.
    @Override
    public void releaseGpuResources() { }

    // ── KV cache adapter wiring ───────────────────────────────────────────────

    public void setKvAdapter(NodeKVCacheAdapter adapter) {
        this.kvAdapter = adapter;
    }

    public void evict(String requestId) {
        kvCacheK.remove(requestId);
        kvCacheV.remove(requestId);
        NodeKVCacheAdapter a = kvAdapter;
        if (a != null) a.evict(requestId);
    }

    // ── Initial activation ────────────────────────────────────────────────────

    private float[] getInitialActivation(ForwardRequest request) {
        if (hasEmbeddings && request.isFirstNode()) {
            int[] tokenIds = request.tokenIds();
            int tokenId = tokenIds[tokenIds.length - 1];
            tokenId = Math.max(0, Math.min(tokenId, cfg.vocabSize() - 1));
            int H = cfg.hiddenDim();
            float[] x = new float[H];
            System.arraycopy(tokenEmbd, tokenId * H, x, 0, H);
            return x;
        }
        float[] src = request.activations();
        return Arrays.copyOf(src, src.length);
    }

    // ── Layer loop ────────────────────────────────────────────────────────────

    private float[] runLayers(float[] x, String requestId, int pos) {
        int L     = endLayer - startLayer;
        int kvDim = cfg.kvDim();

        boolean isNew = kvCacheK.putIfAbsent(requestId,
                new float[L][INITIAL_SEQ_CAPACITY * kvDim]) == null;
        kvCacheV.computeIfAbsent(requestId, k -> new float[L][INITIAL_SEQ_CAPACITY * kvDim]);

        float[][] kCache = kvCacheK.get(requestId);
        float[][] vCache = kvCacheV.get(requestId);

        NodeKVCacheAdapter a = kvAdapter;
        if (isNew && pos > 0 && a != null) {
            for (int li = 0; li < L; li++) {
                int absLayer = startLayer + li;
                final int idx = li;
                a.tryRestore(requestId, absLayer, kvDim).ifPresent(pair -> {
                    ensureKvCapacity(kCache, pair.k().length / kvDim - 1, kvDim);
                    ensureKvCapacity(vCache, pair.v().length / kvDim - 1, kvDim);
                    System.arraycopy(pair.k(), 0, kCache[idx], 0, pair.k().length);
                    System.arraycopy(pair.v(), 0, vCache[idx], 0, pair.v().length);
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

    // ── Single transformer layer ──────────────────────────────────────────────

    /**
     * Phi-2 parallel attention + FFN.
     *
     * <p>Unlike LLaMA's sequential {@code x = x + attn(norm1(x))} then
     * {@code x = x + ffn(norm2(x))}, Phi-2 runs both sub-layers on the
     * <em>same</em> normalised input and merges them in one step:
     * <pre>
     *   xNorm   = layerNorm(x, attn_norm_w, attn_norm_b)
     *   attnOut = attn_output.weight @ attention(xNorm) + attn_output_bias
     *   ffnOut  = ffn_down @ gelu(ffn_up @ xNorm + ffn_up_bias) + ffn_down_bias
     *   x       = x + attnOut + ffnOut
     * </pre>
     */
    private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
        int H     = cfg.hiddenDim();
        int kvDim = cfg.kvDim();

        // Shared LayerNorm for both paths.
        float[] xNorm = layerNorm(x, attnNorm[li], attnNormBias[li], cfg.rmsNormEps());

        // ── Attention path ────────────────────────────────────────────────────
        // Fused QKV: rows [0, H) → Q, [H, H+kvDim) → K, [H+kvDim, end) → V.
        float[] q = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, 0,           H,           H);
        float[] k = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, H,           H + kvDim,   H);
        float[] v = LlamaTransformerHandler.matVec(attnQkv[li], xNorm, H + kvDim,   H + 2*kvDim, H);

        if (attnQkvBias[li] != null) {
            float[] b = attnQkvBias[li];
            for (int i = 0; i < H;     i++) q[i] += b[i];
            for (int i = 0; i < kvDim; i++) k[i] += b[H + i];
            for (int i = 0; i < kvDim; i++) v[i] += b[H + kvDim + i];
        }

        // Partial RoPE: only the first ropeDim dims of each head are rotated.
        ropePartial(q, pos, cfg.numHeads(),   cfg.headDim(), ropeDim, cfg.ropeTheta());
        ropePartial(k, pos, cfg.numKvHeads(), cfg.headDim(), ropeDim, cfg.ropeTheta());

        System.arraycopy(k, 0, kCacheLayer, pos * kvDim, kvDim);
        System.arraycopy(v, 0, vCacheLayer, pos * kvDim, kvDim);

        float[] attnOut  = gqa(q, kCacheLayer, vCacheLayer, pos + 1);
        float[] attnProj = LlamaTransformerHandler.matVec(wo[li], attnOut, H, H);
        if (woBias[li] != null)
            for (int i = 0; i < H; i++) attnProj[i] += woBias[li][i];

        // ── FFN path ──────────────────────────────────────────────────────────
        float[] ffnOut = ffn(xNorm, li);

        // ── Single residual: x + attn_proj + ffn_out ─────────────────────────
        float[] result = new float[H];
        for (int i = 0; i < H; i++)
            result[i] = x[i] + attnProj[i] + ffnOut[i];
        return result;
    }

    /**
     * Phi-2 FFN: {@code gelu(up(xNorm)) → down}, no gate tensor.
     * Both up and down projections carry optional bias vectors.
     */
    private float[] ffn(float[] xNorm, int li) {
        int H = cfg.hiddenDim();
        int I = cfg.intermediateSize();

        float[] up = LlamaTransformerHandler.matVec(wUp[li], xNorm, I, H);
        if (wUpBias[li] != null)
            for (int i = 0; i < I; i++) up[i] += wUpBias[li][i];

        float[] hidden = new float[I];
        for (int i = 0; i < I; i++)
            hidden[i] = gelu(up[i]);

        float[] down = LlamaTransformerHandler.matVec(wDown[li], hidden, H, I);
        if (wDownBias[li] != null)
            for (int i = 0; i < H; i++) down[i] += wDownBias[li][i];

        return down;
    }

    /**
     * Final LayerNorm + output projection → float[vocabSize] logits.
     * Applies optional output bias if the GGUF contains {@code output.bias}.
     */
    private float[] outputProjection(float[] x) {
        float[] xNorm  = layerNorm(x, outputNorm, outputNormBias, cfg.rmsNormEps());
        // Use cfg.vocabSize() — LlamaConfig already picks max(arch, tokenizer vocab).
        float[] logits = LlamaTransformerHandler.matVec(outputProj, xNorm, cfg.vocabSize(), cfg.hiddenDim());
        if (outputBias != null)
            for (int i = 0; i < logits.length; i++) logits[i] += outputBias[i];
        return logits;
    }

    // ── GQA (identical logic to LlamaTransformerHandler) ─────────────────────

    private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
        int H    = cfg.numHeads();
        int Hd   = cfg.headDim();
        int gqaR = cfg.gqaRatio();
        float scale  = (float)(1.0 / Math.sqrt(Hd));
        float[] out    = new float[H * Hd];
        float[] scores = new float[seqLen];

        for (int h = 0; h < H; h++) {
            int kvHead = h / gqaR;
            int qBase  = h * Hd;
            int kBase  = kvHead * Hd;

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

    // ── Math primitives ───────────────────────────────────────────────────────

    /**
     * Standard LayerNorm: {@code y = (x − mean) / sqrt(var + ε) * weight + bias}.
     *
     * <p>Phi-2 uses LayerNorm (not RMSNorm). The {@code bias} array is optional;
     * pass {@code null} to skip the additive bias term.
     */
    static float[] layerNorm(float[] x, float[] weight, float[] bias, float eps) {
        int n = x.length;
        float mean = 0f;
        for (float v : x) mean += v;
        mean /= n;
        float var = 0f;
        for (float v : x) { float d = v - mean; var += d * d; }
        var /= n;
        float scale = 1f / (float) Math.sqrt(var + eps);
        float[] out = new float[n];
        if (bias != null) {
            for (int i = 0; i < n; i++)
                out[i] = weight[i] * (x[i] - mean) * scale + bias[i];
        } else {
            for (int i = 0; i < n; i++)
                out[i] = weight[i] * (x[i] - mean) * scale;
        }
        return out;
    }

    /**
     * Partial RoPE for Phi-2: rotates only the first {@code ropeDim} dimensions
     * of each head; the remaining {@code headDim - ropeDim} dimensions are unchanged.
     *
     * <p>Uses adjacent-pair convention {@code (x[2i], x[2i+1])} — the same as
     * {@link LlamaTransformerHandler#rope} — since llama.cpp's converter pre-permutes
     * both LLaMA and Phi-2 weights to this layout.
     *
     * <p>Frequencies are computed with {@code ropeDim} as the denominator (not
     * {@code headDim}), matching llama.cpp's {@code ggml_rope_ext} for phi2.
     */
    static void ropePartial(float[] x, int pos, int nHeads, int headDim,
                             int ropeDim, float ropeTheta) {
        int halfRope = ropeDim / 2;
        for (int h = 0; h < nHeads; h++) {
            int base = h * headDim;
            for (int i = 0; i < halfRope; i++) {
                double freq  = 1.0 / Math.pow(ropeTheta, (2.0 * i) / ropeDim);
                double angle = pos * freq;
                float  cosA  = (float) Math.cos(angle);
                float  sinA  = (float) Math.sin(angle);
                float  x0    = x[base + 2*i];
                float  x1    = x[base + 2*i + 1];
                x[base + 2*i]     = x0 * cosA - x1 * sinA;
                x[base + 2*i + 1] = x0 * sinA + x1 * cosA;
            }
            // Dims [ropeDim .. headDim-1] are intentionally left unchanged.
        }
    }

    /**
     * GELU activation used by Phi-2's FFN.
     *
     * <p>Uses the standard tanh approximation:
     * {@code gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))}
     * which matches PyTorch's {@code torch.nn.functional.gelu} default.
     */
    static float gelu(float x) {
        double xd = x;
        return (float)(0.5 * xd * (1.0 + Math.tanh(
                Math.sqrt(2.0 / Math.PI) * (xd + 0.044715 * xd * xd * xd))));
    }
}
