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

package cab.ml.juno.vision;

import java.io.IOException;
import java.nio.file.Path;
import java.util.logging.Logger;

import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.MatVec;

/**
 * Pure Java CLIP ViT-L/14 encoder.
 *
 * Reads pre-trained CLIP weights from a GGUF file that follows the
 * llama.cpp mmproj (multimodal projector) naming convention used by
 * LLaVA-1.5 and Phi-3 Vision:
 *
 * <pre>
 *   v.patch_embd.weight          [hiddenSize, 3 * patchSize * patchSize]
 *   v.patch_embd.bias            [hiddenSize]
 *   v.position_embd.weight       [numVisionTokens, hiddenSize]   (CLS included)
 *   v.class_embd                 [hiddenSize]
 *   v.pre_ln.weight / .bias      [hiddenSize]
 *   v.blk.{i}.attn_q.weight / .bias
 *   v.blk.{i}.attn_k.weight / .bias
 *   v.blk.{i}.attn_v.weight / .bias
 *   v.blk.{i}.attn_out.weight / .bias
 *   v.blk.{i}.ln1.weight / .bias
 *   v.blk.{i}.ln2.weight / .bias
 *   v.blk.{i}.ffn_up.weight / .bias
 *   v.blk.{i}.ffn_down.weight / .bias
 *   mm.0.weight / mm.0.bias      [projectionDim, hiddenSize]  — vision projector
 * </pre>
 *
 * Forward pass:
 * <ol>
 *   <li>Patch embedding: each image patch → hiddenSize vector via linear.
 *   <li>Prepend CLS token embedding.
 *   <li>Add position embeddings.
 *   <li>Pre-encoder layer norm.
 *   <li>N CLIP transformer blocks (LayerNorm → self-attention → LayerNorm → MLP).
 *   <li>Vision projector (single linear, GELU optional): maps hiddenSize →
 *       projectionDim (= LLM hiddenDim).
 *   <li>Return patch embeddings excluding CLS (shape: numPatches × projectionDim).
 * </ol>
 *
 * The output float[][] is directly consumed by {@link VisionAwareForwardPassHandler}
 * to splice vision tokens into the LLM's residual stream.
 *
 * Thread-safe after construction — all weights are read-only.
 */
public final class VisionEncoder {

    private static final Logger log = Logger.getLogger(VisionEncoder.class.getName());

    private final VisionConfig cfg;
    private final MatVec backend;

    // ── Patch & position embeddings ────────────────────────────────────────
    private final float[] patchEmbdWeight;  // [hiddenSize, 3 * patchSize * patchSize]
    private final float[] patchEmbdBias;    // [hiddenSize]
    private final float[] posEmbd;          // [numVisionTokens * hiddenSize]
    private final float[] classEmbd;        // [hiddenSize]

    // ── Pre-encoder layer norm ─────────────────────────────────────────────
    private final float[] preLnWeight;      // [hiddenSize]
    private final float[] preLnBias;        // [hiddenSize]

    // ── Per-layer weights (L layers) ───────────────────────────────────────
    private final float[][] ln1Weight;      // [L][hiddenSize]
    private final float[][] ln1Bias;        // [L][hiddenSize]
    private final float[][] ln2Weight;      // [L][hiddenSize]
    private final float[][] ln2Bias;        // [L][hiddenSize]
    private final float[][] wq;             // [L][hiddenSize * hiddenSize]
    private final float[][] bq;             // [L][hiddenSize]
    private final float[][] wk;             // [L][hiddenSize * hiddenSize]
    private final float[][] bk;             // [L][hiddenSize]
    private final float[][] wv;             // [L][hiddenSize * hiddenSize]
    private final float[][] bv;             // [L][hiddenSize]
    private final float[][] wOut;           // [L][hiddenSize * hiddenSize]
    private final float[][] bOut;           // [L][hiddenSize]
    private final float[][] ffnUp;          // [L][intermediateSize * hiddenSize]
    private final float[][] bffnUp;         // [L][intermediateSize]
    private final float[][] ffnDown;        // [L][hiddenSize * intermediateSize]
    private final float[][] bffnDown;       // [L][hiddenSize]

    // ── Vision projector ──────────────────────────────────────────────────
    private final float[] projWeight;       // [projectionDim * hiddenSize]
    private final float[] projBias;         // [projectionDim]  — null when absent

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Load vision encoder weights from an open GgufReader.
     *
     * @param r       open reader; not closed by this method
     * @param cfg     parsed vision configuration
     * @param backend MatVec backend to use for matrix multiplies (CPU or GPU)
     */
    public static VisionEncoder load(GgufReader r, VisionConfig cfg, MatVec backend) throws IOException {
        log.info("Loading vision encoder: " + cfg);
        return new VisionEncoder(r, cfg, backend);
    }

    /**
     * Load vision encoder weights from a GGUF file by path.
     *
     * @param modelPath path to the .gguf file containing vision weights
     * @param backend   MatVec backend
     */
    public static VisionEncoder load(Path modelPath, MatVec backend) throws IOException {
        try (GgufReader r = GgufReader.open(modelPath)) {
            VisionConfig cfg = VisionConfig.from(r);
            return new VisionEncoder(r, cfg, backend);
        }
    }

    private VisionEncoder(GgufReader r, VisionConfig cfg, MatVec backend) throws IOException {
        this.cfg = cfg;
        this.backend = backend;
        int L = cfg.numLayers();
        int H = cfg.hiddenSize();
        int I = cfg.intermediateSize();
        int P = cfg.projectionDim();
        int patchElems = 3 * cfg.patchSize() * cfg.patchSize();

        patchEmbdWeight = r.tensor("v.patch_embd.weight");   // H × patchElems
        patchEmbdBias   = r.hasTensor("v.patch_embd.bias")
                        ? r.tensor("v.patch_embd.bias") : new float[H];
        posEmbd         = r.tensor("v.position_embd.weight"); // numVisionTokens × H
        classEmbd       = r.tensor("v.class_embd");           // H

        preLnWeight = r.hasTensor("v.pre_ln.weight") ? r.tensor("v.pre_ln.weight") : onesF(H);
        preLnBias   = r.hasTensor("v.pre_ln.bias")   ? r.tensor("v.pre_ln.bias")   : new float[H];

        ln1Weight = new float[L][];
        ln1Bias   = new float[L][];
        ln2Weight = new float[L][];
        ln2Bias   = new float[L][];
        wq = new float[L][]; bq = new float[L][];
        wk = new float[L][]; bk = new float[L][];
        wv = new float[L][]; bv = new float[L][];
        wOut = new float[L][]; bOut = new float[L][];
        ffnUp = new float[L][]; bffnUp = new float[L][];
        ffnDown = new float[L][]; bffnDown = new float[L][];

        for (int i = 0; i < L; i++) {
            String p = "v.blk." + i + ".";
            ln1Weight[i] = r.tensor(p + "ln1.weight");
            ln1Bias[i]   = r.hasTensor(p + "ln1.bias") ? r.tensor(p + "ln1.bias") : new float[H];
            ln2Weight[i] = r.tensor(p + "ln2.weight");
            ln2Bias[i]   = r.hasTensor(p + "ln2.bias") ? r.tensor(p + "ln2.bias") : new float[H];
            wq[i]   = r.tensor(p + "attn_q.weight");
            bq[i]   = r.hasTensor(p + "attn_q.bias")   ? r.tensor(p + "attn_q.bias")   : new float[H];
            wk[i]   = r.tensor(p + "attn_k.weight");
            bk[i]   = r.hasTensor(p + "attn_k.bias")   ? r.tensor(p + "attn_k.bias")   : new float[H];
            wv[i]   = r.tensor(p + "attn_v.weight");
            bv[i]   = r.hasTensor(p + "attn_v.bias")   ? r.tensor(p + "attn_v.bias")   : new float[H];
            wOut[i] = r.tensor(p + "attn_out.weight");
            bOut[i] = r.hasTensor(p + "attn_out.bias") ? r.tensor(p + "attn_out.bias") : new float[H];
            ffnUp[i]   = r.tensor(p + "ffn_up.weight");
            bffnUp[i]  = r.hasTensor(p + "ffn_up.bias")   ? r.tensor(p + "ffn_up.bias")   : new float[I];
            ffnDown[i] = r.tensor(p + "ffn_down.weight");
            bffnDown[i]= r.hasTensor(p + "ffn_down.bias") ? r.tensor(p + "ffn_down.bias") : new float[H];
        }

        projWeight = r.tensor("mm.0.weight");              // P × H
        projBias   = r.hasTensor("mm.0.bias") ? r.tensor("mm.0.bias") : null;

        log.info("Vision encoder loaded — " + L + " blocks, hidden=" + H
                + " patches=" + cfg.numPatches() + " proj=" + P);
    }

    // ── Public API ────────────────────────────────────────────────────────

    /** Parsed configuration. */
    public VisionConfig config() {
        return cfg;
    }

    /**
     * Encode pixel tensor to patch embeddings in LLM token space.
     *
     * @param pixelTensor float[3 * imageSize * imageSize] CHW, CLIP-normalised
     * @return float[numPatches][projectionDim] — one embedding per image patch
     *         in raster order (left-to-right, top-to-bottom); CLS excluded.
     */
    public float[][] encode(float[] pixelTensor) {
        int N  = cfg.numVisionTokens();  // numPatches + 1 CLS
        int H  = cfg.hiddenSize();
        int nP = cfg.numPatches();

        // Step 1 — patch embedding: linear projection of each raw patch
        float[][] tokens = patchEmbed(pixelTensor, H, nP);

        // Step 2 — prepend CLS token
        float[][] withCls = new float[N][H];
        System.arraycopy(classEmbd, 0, withCls[0], 0, H);
        for (int i = 0; i < nP; i++)
            withCls[i + 1] = tokens[i];

        // Step 3 — add position embeddings
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < H; d++)
                withCls[i][d] += posEmbd[i * H + d];
        }

        // Step 4 — pre-encoder layer norm
        for (int i = 0; i < N; i++)
            withCls[i] = layerNorm(withCls[i], preLnWeight, preLnBias, cfg.layerNormEps());

        // Step 5 — N CLIP transformer blocks
        for (int li = 0; li < cfg.numLayers(); li++)
            withCls = transformerBlock(withCls, li, N, H);

        // Step 6 — vision projector on patch tokens only (drop CLS)
        float[][] out = new float[nP][];
        for (int i = 0; i < nP; i++)
            out[i] = project(withCls[i + 1]);

        return out;
    }

    // ── Patch embedding ───────────────────────────────────────────────────

    /**
     * Map each image patch to a hiddenSize vector via a learned linear transform.
     *
     * The pixel tensor is in CHW (channel-first) order. Each patch is extracted
     * column-by-column across the three channels, then multiplied by
     * {@code patchEmbdWeight}.
     */
    private float[][] patchEmbed(float[] pixelTensor, int H, int nP) {
        int pSz = cfg.patchSize();
        int imgW = cfg.imageSize();
        int patchElems = 3 * pSz * pSz;
        int gridW = imgW / pSz;

        float[][] out = new float[nP][H];
        float[] patch = new float[patchElems];

        for (int py = 0; py < gridW; py++) {
            for (int px = 0; px < gridW; px++) {
                // Extract patch pixels in CHW order
                for (int c = 0; c < 3; c++) {
                    int planeBase = c * imgW * imgW;
                    for (int dy = 0; dy < pSz; dy++) {
                        for (int dx = 0; dx < pSz; dx++) {
                            int pixIdx = planeBase + (py * pSz + dy) * imgW + (px * pSz + dx);
                            patch[c * pSz * pSz + dy * pSz + dx] = pixelTensor[pixIdx];
                        }
                    }
                }
                int patchIdx = py * gridW + px;
                float[] emb = backend.sgemv(patchEmbdWeight, patch, H, patchElems);
                for (int d = 0; d < H; d++)
                    out[patchIdx][d] = emb[d] + patchEmbdBias[d];
            }
        }
        return out;
    }

    // ── CLIP transformer block ────────────────────────────────────────────

    private float[][] transformerBlock(float[][] x, int li, int N, int H) {
        // Self-attention sub-layer with pre-LayerNorm
        float[][] xNorm1 = new float[N][];
        for (int i = 0; i < N; i++)
            xNorm1[i] = layerNorm(x[i], ln1Weight[li], ln1Bias[li], cfg.layerNormEps());

        float[][] attnOut = selfAttention(xNorm1, li, N, H);

        // Residual
        float[][] x2 = new float[N][H];
        for (int i = 0; i < N; i++)
            for (int d = 0; d < H; d++)
                x2[i][d] = x[i][d] + attnOut[i][d];

        // MLP sub-layer with pre-LayerNorm
        float[][] xNorm2 = new float[N][];
        for (int i = 0; i < N; i++)
            xNorm2[i] = layerNorm(x2[i], ln2Weight[li], ln2Bias[li], cfg.layerNormEps());

        float[][] mlpOut = new float[N][];
        for (int i = 0; i < N; i++)
            mlpOut[i] = mlp(xNorm2[i], li);

        // Residual
        float[][] x3 = new float[N][H];
        for (int i = 0; i < N; i++)
            for (int d = 0; d < H; d++)
                x3[i][d] = x2[i][d] + mlpOut[i][d];

        return x3;
    }

    // ── Self-attention (SDPA, no causal mask) ─────────────────────────────

    private float[][] selfAttention(float[][] x, int li, int N, int H) {
        int nH  = cfg.numHeads();
        int dH  = cfg.headDim();

        // Project Q, K, V for all tokens
        float[][] Q = new float[N][H];
        float[][] K = new float[N][H];
        float[][] V = new float[N][H];
        for (int i = 0; i < N; i++) {
            float[] q = backend.sgemv(wq[li], x[i], H, H);
            float[] k = backend.sgemv(wk[li], x[i], H, H);
            float[] v = backend.sgemv(wv[li], x[i], H, H);
            for (int d = 0; d < H; d++) {
                Q[i][d] = q[d] + bq[li][d];
                K[i][d] = k[d] + bk[li][d];
                V[i][d] = v[d] + bv[li][d];
            }
        }

        // Scaled dot-product attention per head (no causal mask — full attention)
        float scale = 1.0f / (float) Math.sqrt(dH);
        float[][] attnOut = new float[N][H];

        for (int h = 0; h < nH; h++) {
            int hOff = h * dH;
            // scores[i][j] = (Q[i][h*dH..] · K[j][h*dH..]) * scale
            float[][] scores = new float[N][N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float dot = 0f;
                    for (int d = 0; d < dH; d++)
                        dot += Q[i][hOff + d] * K[j][hOff + d];
                    scores[i][j] = dot * scale;
                }
                // Softmax over j
                softmaxInPlace(scores[i]);
                // Weighted sum of V
                for (int j = 0; j < N; j++) {
                    float w = scores[i][j];
                    for (int d = 0; d < dH; d++)
                        attnOut[i][hOff + d] += w * V[j][hOff + d];
                }
            }
        }

        // Output projection
        float[][] out = new float[N][H];
        for (int i = 0; i < N; i++) {
            float[] proj = backend.sgemv(wOut[li], attnOut[i], H, H);
            for (int d = 0; d < H; d++)
                out[i][d] = proj[d] + bOut[li][d];
        }
        return out;
    }

    // ── MLP (GELU activation) ─────────────────────────────────────────────

    private float[] mlp(float[] x, int li) {
        int I = cfg.intermediateSize();
        int H = cfg.hiddenSize();
        float[] hidden = backend.sgemv(ffnUp[li], x, I, H);
        for (int i = 0; i < I; i++)
            hidden[i] = gelu(hidden[i] + bffnUp[li][i]);
        float[] out = backend.sgemv(ffnDown[li], hidden, H, I);
        for (int d = 0; d < H; d++)
            out[d] += bffnDown[li][d];
        return out;
    }

    // ── Vision projector (single linear: hiddenSize → projectionDim) ──────

    private float[] project(float[] x) {
        int P = cfg.projectionDim();
        int H = cfg.hiddenSize();
        float[] out = backend.sgemv(projWeight, x, P, H);
        if (projBias != null)
            for (int i = 0; i < P; i++)
                out[i] += projBias[i];
        return out;
    }

    // ── Math primitives ───────────────────────────────────────────────────

    static float[] layerNorm(float[] x, float[] weight, float[] bias, float eps) {
        int n = x.length;
        float mean = 0f;
        for (float v : x) mean += v;
        mean /= n;
        float var = 0f;
        for (float v : x) { float d = v - mean; var += d * d; }
        var /= n;
        float scale = 1.0f / (float) Math.sqrt(var + eps);
        float[] out = new float[n];
        for (int i = 0; i < n; i++)
            out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
        return out;
    }

    /** Gaussian Error Linear Unit — tanh approximation used by CLIP. */
    static float gelu(float x) {
        // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c = 0.7978845608f; // sqrt(2/pi)
        float t = (float) Math.tanh(c * (x + 0.044715f * x * x * x));
        return 0.5f * x * (1f + t);
    }

    private static void softmaxInPlace(float[] x) {
        float max = x[0];
        for (float v : x) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < x.length; i++) { x[i] = (float) Math.exp(x[i] - max); sum += x[i]; }
        for (int i = 0; i < x.length; i++) x[i] /= sum;
    }

    private static float[] onesF(int n) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, 1.0f);
        return a;
    }
}