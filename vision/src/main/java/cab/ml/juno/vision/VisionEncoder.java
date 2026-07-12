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
import java.util.Arrays;
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
 *   mm.0.weight / mm.0.bias      [outputDim, hiddenSize]  — vision projector.
 *                                outputDim is read from this tensor's own
 *                                shape, NOT from the (unreliable, see
 *                                {@link #outputDim()}) clip.vision.projection_dim
 *                                metadata field. This is the ONLY projector
 *                                layer actually applied (see below).
 *   mm.2.weight / mm.2.bias      [outputDim, outputDim]   — detected and
 *                                loaded for diagnostic logging only. Applying
 *                                it (GELU between mm.0 and mm.2) was tried and
 *                                reverted 2026-07-12: it caused a confirmed
 *                                regression (degenerate repeating-token output)
 *                                rather than an improvement. Do not re-enable
 *                                without new evidence of the actual root cause.
 * </pre>
 *
 * Forward pass:
 * <ol>
 *   <li>Patch embedding: each image patch → hiddenSize vector via linear.
 *   <li>Prepend CLS token embedding.
 *   <li>Add position embeddings.
 *   <li>Pre-encoder layer norm.
 *   <li>N CLIP transformer blocks (LayerNorm → self-attention → LayerNorm → MLP).
 *   <li>Vision projector: mm.0 only (hiddenSize → outputDim). mm.2, when
 *       present in the file, is loaded and logged but intentionally not
 *       applied — see the mm.2 note above.
 *   <li>Return patch embeddings excluding CLS (shape: numPatches × outputDim).
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
    private final float[] projWeight;       // [outputDim * hiddenSize]  mm.0
    private final float[] projBias;         // [outputDim]  — null when absent
    private final float[] projWeight2;      // [outputDim * outputDim]  mm.2 — null for single-layer projectors
    private final float[] projBias2;        // [outputDim]  — null when mm.2 absent or has no bias
    private final int outputDim;            // actual projector output width (see #outputDim())

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
            loadFfn(r, p, i, H, I);
        }

        projWeight = r.tensor("mm.0.weight");
        // clip.vision.projection_dim metadata is not reliable across mmproj exports
        // (same lesson as loadFfn): this file declares 768 but the tensor's own
        // shape is actually [hiddenSize, 3072] — 3072 being the LLM's own hidden
        // dimension, the real width the projector must produce to be spliced into
        // the LLM's embedding space. Derive it from the tensor itself, not metadata.
        long[] projDims = r.tensorDims("mm.0.weight");
        this.outputDim = resolveProjectorOutputDim(projDims[0], projDims[projDims.length - 1], H,
                projWeight.length, cfg.projectionDim());

        projBias = r.hasTensor("mm.0.bias") ? r.tensor("mm.0.bias") : null;
        if (projBias != null && projBias.length != this.outputDim) {
            throw new IllegalStateException("Vision projector mm.0.bias has length " + projBias.length
                    + ", expected outputDim=" + this.outputDim);
        }

        // LLaVA-1.5's standard mm_projector_type is "mlp2x_gelu": mm.0 (hiddenSize→
        // outputDim) → GELU → mm.2 (outputDim→outputDim). llama.cpp mmproj GGUF
        // files name the layers mm.0/mm.2 (mm.1 is the implicit, weight-less GELU).
        //
        // 2026-07-12 UPDATE: mm.2 IS present in this specific mmproj file and WAS
        // wired up to apply GELU→mm.2 after mm.0, on the theory that a single
        // mm.0 linear layer alone is not what llava-v1.5-7b was trained with.
        // That theory is WRONG, or at least this implementation of it is: it was
        // confirmed by the user to be a regression — output degenerated from
        // "coherent but hallucinated" to a repeating <image>-token loop with no
        // real content, which is the signature of a numerically broken embedding
        // rather than merely an incomplete transform. mm.2 is still loaded and
        // validated here (so this diagnostic information stays visible in the
        // log) but is deliberately NOT applied in project() below. Root cause of
        // why applying it corrupts the output is not yet understood — do not
        // re-enable without new evidence (e.g. dumping actual patch-vector
        // magnitudes/NaN checks before vs after the mm.2 step). See CHANGELOG.
        if (r.hasTensor("mm.2.weight")) {
            long[] proj2Dims = r.tensorDims("mm.2.weight");
            if (proj2Dims[0] != this.outputDim || proj2Dims[proj2Dims.length - 1] != this.outputDim) {
                throw new IllegalStateException("Vision projector mm.2.weight has shape " + Arrays.toString(proj2Dims)
                        + ", expected [" + this.outputDim + ", " + this.outputDim + "] (outputDim → outputDim)");
            }
            projWeight2 = r.tensor("mm.2.weight");
            projBias2 = r.hasTensor("mm.2.bias") ? r.tensor("mm.2.bias") : null;
            if (projBias2 != null && projBias2.length != this.outputDim) {
                throw new IllegalStateException("Vision projector mm.2.bias has length " + projBias2.length
                        + ", expected outputDim=" + this.outputDim);
            }
            log.info("Vision projector: mm.2.weight IS present (outputDim=" + this.outputDim + ") but is NOT "
                    + "applied — 2026-07-12 regression, see VisionEncoder javadoc/CHANGELOG. Using mm.0 only.");
        } else {
            projWeight2 = null;
            projBias2 = null;
            log.info("Vision projector: mm.2.weight not found. Using mm.0 only (single linear layer).");
        }

        log.info("Vision encoder loaded — " + L + " blocks, hidden=" + H
                + " patches=" + cfg.numPatches() + " outputDim=" + this.outputDim);
    }

    /**
     * Actual width of the vision projector's output — i.e. the dimension of
     * each patch vector returned by {@link #encode}.
     *
     * This is derived from {@code mm.0.weight}'s own GGUF shape, NOT from
     * {@link VisionConfig#projectionDim()}: {@code clip.vision.projection_dim}
     * metadata is not reliable across mmproj exports (some files, including
     * llava-phi-3-mini's, declare CLIP's own native projection width there
     * rather than the actual mm-projector output width used to splice into
     * the LLM's embedding space). Callers that need the dimension patch
     * vectors will actually have — e.g. to size
     * {@code VisionAwareForwardPassHandler}'s {@code hiddenDim} — must use
     * this method, not {@code config().projectionDim()}.
     */
    public int outputDim() {
        return outputDim;
    }

    /**
     * Pure decision logic (no I/O): validates {@code mm.0.weight}'s own
     * measured shape against {@code hiddenSize} and returns the real output
     * dimension, logging a warning (not an error — the tensor shape, not the
     * metadata, wins) if it disagrees with the unreliable
     * {@code clip.vision.projection_dim} metadata value.
     *
     * Package-private and static so it is directly unit-testable without
     * constructing a synthetic GGUF file.
     *
     * @param inDim              mm.0.weight's declared input dimension (first GGUF dim)
     * @param outDim             mm.0.weight's declared output dimension (last GGUF dim)
     * @param hiddenSize         the vision encoder's hidden size (expected inDim)
     * @param weightLength       mm.0.weight's actual flattened element count
     * @param metadataProjection {@code clip.vision.projection_dim} as read from GGUF metadata
     * @throws IllegalStateException if inDim doesn't match hiddenSize, or the
     *         flattened weight length doesn't match outDim * hiddenSize
     */
    static int resolveProjectorOutputDim(long inDim, long outDim, int hiddenSize, long weightLength,
            int metadataProjection) {
        if (inDim != hiddenSize) {
            throw new IllegalStateException("Vision projector mm.0.weight has inDim=" + inDim
                    + ", expected hiddenSize=" + hiddenSize + " — check clip.vision.embedding_length metadata.");
        }
        int resolvedOutputDim = Math.toIntExact(outDim);
        if ((long) resolvedOutputDim * hiddenSize != weightLength) {
            throw new IllegalStateException("Vision projector mm.0.weight has " + weightLength
                    + " elements, expected outputDim(" + resolvedOutputDim + ") * hiddenSize(" + hiddenSize + ")="
                    + ((long) resolvedOutputDim * hiddenSize));
        }
        if (resolvedOutputDim != metadataProjection) {
            log.warning("Vision projector's actual output dim (" + resolvedOutputDim
                    + ", from mm.0.weight's own shape) does not match clip.vision.projection_dim metadata ("
                    + metadataProjection + ") — using the tensor's own shape.");
        }
        return resolvedOutputDim;
    }

    /**
     * Loads the two FFN linear layers for block {@code li}, determining which
     * of the two GGUF tensors (named {@code ffn_up}/{@code ffn_down} by
     * convention) is actually the H→I expansion versus the I→H contraction by
     * its own declared output dimension — not by trusting the name.
     *
     * This matters because the {@code ffn_up}/{@code ffn_down} naming
     * convention is not consistently applied across every mmproj GGUF export
     * in the wild: some files' "ffn_up" tensor is in fact the contraction
     * layer. Trusting the name blindly loads a bias vector of the wrong
     * length into the wrong slot, which does not fail until deep in a forward
     * pass — as an opaque {@code ArrayIndexOutOfBoundsException} with no
     * indication of which tensor or model was at fault. Reading each tensor's
     * own {@link GgufReader#tensorDims} up front turns that into an immediate,
     * descriptive failure at load time instead.
     */
    private void loadFfn(GgufReader r, String p, int li, int H, int I) throws IOException {
        String upWeightName = p + "ffn_up.weight";
        String downWeightName = p + "ffn_down.weight";
        float[] upWeight = r.tensor(upWeightName);
        float[] downWeight = r.tensor(downWeightName);

        // GGUF stores 2D weights as [inDim, outDim] (innermost dim first) —
        // see GgufReader.tensorDims javadoc. outDim is the last entry.
        long[] upDims = r.tensorDims(upWeightName);
        long[] downDims = r.tensorDims(downWeightName);
        long upOutDim = upDims[upDims.length - 1];
        long downOutDim = downDims[downDims.length - 1];

        FfnOrientation orientation = resolveFfnOrientation(li, upWeightName, downWeightName, upOutDim, downOutDim, I,
                H);

        float[] expandWeight = orientation == FfnOrientation.NORMAL ? upWeight : downWeight;
        float[] contractWeight = orientation == FfnOrientation.NORMAL ? downWeight : upWeight;
        String expandBiasName = orientation == FfnOrientation.NORMAL ? p + "ffn_up.bias" : p + "ffn_down.bias";
        String contractBiasName = orientation == FfnOrientation.NORMAL ? p + "ffn_down.bias" : p + "ffn_up.bias";

        if (orientation == FfnOrientation.SWAPPED) {
            log.warning("Vision encoder block " + li + ": ffn_up/ffn_down tensor names are reversed relative to "
                    + "the usual expand/contract convention in this mmproj file (ffn_up outDim=" + upOutDim
                    + ", expected I=" + I + ") — using each tensor's own shape rather than its name.");
        }

        ffnUp[li]    = expandWeight;
        bffnUp[li]   = r.hasTensor(expandBiasName) ? r.tensor(expandBiasName) : new float[I];
        ffnDown[li]  = contractWeight;
        bffnDown[li] = r.hasTensor(contractBiasName) ? r.tensor(contractBiasName) : new float[H];

        if (bffnUp[li].length != I)
            throw new IllegalStateException("Vision encoder block " + li + ": " + expandBiasName + " has length "
                    + bffnUp[li].length + ", expected intermediateSize=" + I);
        if (bffnDown[li].length != H)
            throw new IllegalStateException("Vision encoder block " + li + ": " + contractBiasName + " has length "
                    + bffnDown[li].length + ", expected hiddenSize=" + H);
    }

    /** Which of the two named FFN tensors is actually the H→I expansion. */
    enum FfnOrientation {
        /** {@code ffn_up} is the H→I expansion, {@code ffn_down} is the I→H contraction (the usual case). */
        NORMAL,
        /** Reversed: {@code ffn_up} is actually I→H and {@code ffn_down} is actually H→I. */
        SWAPPED
    }

    /**
     * Pure decision logic (no I/O): given each FFN weight tensor's measured
     * output dimension, determines whether the file follows the usual
     * ffn_up=expand/ffn_down=contract naming or has it reversed, or throws if
     * neither orientation is consistent with the configured intermediateSize
     * (I) / hiddenSize (H) — i.e. the file's actual architecture does not
     * match the VisionConfig read from its metadata.
     *
     * Package-private and static so it is directly unit-testable without
     * constructing a synthetic GGUF file.
     */
    static FfnOrientation resolveFfnOrientation(int li, String upWeightName, String downWeightName, long upOutDim,
            long downOutDim, int intermediateSize, int hiddenSize) {
        boolean namesMatchConvention = upOutDim == intermediateSize && downOutDim == hiddenSize;
        boolean namesAreSwapped = upOutDim == hiddenSize && downOutDim == intermediateSize;

        if (namesMatchConvention)
            return FfnOrientation.NORMAL;
        if (namesAreSwapped)
            return FfnOrientation.SWAPPED;

        throw new IllegalStateException("Vision encoder block " + li + ": FFN tensor shapes do not match "
                + "either orientation of intermediateSize=" + intermediateSize + " / hiddenSize=" + hiddenSize
                + " — " + upWeightName + " outDim=" + upOutDim + ", " + downWeightName + " outDim=" + downOutDim
                + ". This mmproj file's architecture does not match the VisionConfig read from its metadata; "
                + "check clip.vision.feed_forward_length / clip.vision.embedding_length.");
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
     * @return float[numPatches][outputDim()] — one embedding per image patch
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

    // ── Vision projector: mm.0 -> [GELU -> mm.2] ───────────────────────────

    private float[] project(float[] x) {
        // REVERTED 2026-07-12: applying mm.2 (see load() below) caused a confirmed
        // regression — output degenerated from "coherent but wrong content" to a
        // repeating <image>-token loop (finish_reason=length, no real content at
        // all). That is the signature of a numerically broken embedding, not a
        // "half-applied projector" being merely wrong — something in this specific
        // mm.2 application is actively corrupting the patch vectors, not just
        // failing to complete the intended transform. Root cause not yet found;
        // reverting to the single-linear mm.0-only behavior that was confirmed
        // to at least produce grammatically coherent (if hallucinated) output.
        // See CHANGELOG. projWeight2/projBias2 are intentionally NOT passed here.
        return applyProjector(backend, x, projWeight, projBias, null, null, cfg.hiddenSize(), outputDim);
    }

    /**
     * Pure projector math (no field access): mm.0 linear, then — when
     * {@code w2} is non-null — GELU followed by mm.2 linear.
     *
     * Package-private and static so it is directly unit-testable without
     * constructing a full VisionEncoder from a synthetic GGUF file (same
     * pattern as {@link #resolveProjectorOutputDim}).
     *
     * @param backend    MatVec implementation to multiply with
     * @param x          input vector, length hiddenSize
     * @param w1         mm.0.weight, length outputDim*hiddenSize
     * @param b1         mm.0.bias, length outputDim, or null
     * @param w2         mm.2.weight, length outputDim*outputDim, or null for a
     *                   single-layer projector
     * @param b2         mm.2.bias, length outputDim, or null
     * @param hiddenSize vision encoder hidden size (mm.0's input width)
     * @param outputDim  projector output width (mm.0's output width, and
     *                   mm.2's input/output width when present)
     */
    static float[] applyProjector(MatVec backend, float[] x, float[] w1, float[] b1, float[] w2, float[] b2,
            int hiddenSize, int outputDim) {
        float[] out = backend.sgemv(w1, x, outputDim, hiddenSize);
        if (b1 != null)
            for (int i = 0; i < outputDim; i++)
                out[i] += b1[i];

        if (w2 == null) {
            return out; // single-layer projector (non-standard for llava-v1.5, see load())
        }

        for (int i = 0; i < outputDim; i++)
            out[i] = gelu(out[i]);

        float[] out2 = backend.sgemv(w2, out, outputDim, outputDim);
        if (b2 != null)
            for (int i = 0; i < outputDim; i++)
                out2[i] += b2[i];
        return out2;
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