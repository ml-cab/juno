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
 * Reads pre-trained CLIP or SigLIP weights from a GGUF file following
 * the llama.cpp mmproj (multimodal projector) naming convention used by
 * LLaVA-1.5, Phi-3 Vision, moondream2, and others:
 *
 * <pre>
 *   v.patch_embd.weight          [hiddenSize, 3 * patchSize * patchSize]
 *   v.patch_embd.bias            [hiddenSize]
 *   v.position_embd.weight       [numPositions, hiddenSize]
 *                                numPositions = numPatches+1 for CLIP (CLS included),
 *                                numPositions = numPatches   for SigLIP (no CLS).
 *   v.class_embd                 [hiddenSize]   OPTIONAL — absent for SigLIP models
 *                                (e.g. moondream2). When present, CLIP-style CLS
 *                                token is prepended to the patch sequence.
 *   v.pre_ln.weight / .bias      [hiddenSize]
 *   v.blk.{i}.attn_q.weight / .bias
 *   v.blk.{i}.attn_k.weight / .bias
 *   v.blk.{i}.attn_v.weight / .bias
 *   v.blk.{i}.attn_out.weight / .bias
 *   v.blk.{i}.ln1.weight / .bias
 *   v.blk.{i}.ln2.weight / .bias
 *   v.blk.{i}.ffn_up.weight / .bias
 *   v.blk.{i}.ffn_down.weight / .bias
 *   mm.0.weight / mm.0.bias      [hiddenSize, mm0OutDim] — first projector layer.
 *                                mm0OutDim is read from the tensor's own shape
 *                                (NOT from clip.vision.projection_dim metadata,
 *                                which is unreliable — see {@link #outputDim()}).
 *   mm.2.weight / mm.2.bias      [mm0OutDim, finalOutDim] — second projector layer.
 *                                Applied when mm0OutDim ≠ finalOutDim (moondream2:
 *                                8192→2048, structurally necessary). NOT applied
 *                                when square (mm0OutDim == finalOutDim, LLaVA-1.5
 *                                pattern) — 2026-07-12 regression for that case.
 * </pre>
 *
 * Forward pass:
 * <ol>
 *   <li>Patch embedding: each image patch → hiddenSize vector via linear.
 *   <li>Build sequence via {@link #buildSequence}: prepend CLS when
 *       {@code v.class_embd} is present (CLIP); use patches as-is when absent
 *       (SigLIP / moondream2).
 *   <li>Add position embeddings (length matches the actual sequence length).
 *   <li>Pre-encoder layer norm.
 *   <li>N transformer blocks (LayerNorm → self-attention → LayerNorm → MLP).
 *   <li>Vision projector: mm.0 (hiddenSize → mm0OutDim), then GELU + mm.2
 *       (mm0OutDim → finalOutDim) when structurally necessary (non-square).
 *       Square mm.2 is NOT applied — 2026-07-12 regression; see {@link #project()}.
 *   <li>Return patch embeddings (CLS position skipped when present).
 *       Shape: numPatches × {@link #outputDim()} (= finalOutDim).
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
    private final int mm0OutDim;           // mm.0 output width (= intermediate for 2-layer MLP)
    private final int finalOutDim;          // true output fed to LLM: mm.2 out when applied, else mm0OutDim

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
        posEmbd         = r.tensor("v.position_embd.weight"); // numPositions × H
        // SigLIP encoders (e.g. moondream2) have no CLS token — v.class_embd absent.
        classEmbd       = r.hasTensor("v.class_embd") ? r.tensor("v.class_embd") : null;

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
        // mm.0 output dim is read from the tensor's own shape — clip.vision.projection_dim
        // metadata is not reliable (see resolveProjectorOutputDim javadoc and CHANGELOG).
        long[] projDims = r.tensorDims("mm.0.weight");
        this.mm0OutDim = resolveProjectorOutputDim(projDims[0], projDims[projDims.length - 1], H,
                projWeight.length, cfg.projectionDim());

        projBias = r.hasTensor("mm.0.bias") ? r.tensor("mm.0.bias") : null;
        if (projBias != null && projBias.length != this.mm0OutDim)
            throw new IllegalStateException("Vision projector mm.0.bias has length " + projBias.length
                    + ", expected mm0OutDim=" + this.mm0OutDim);

        // mm.2 (the second projector layer) is handled differently depending on
        // its shape relative to mm.0:
        //
        //   NON-SQUARE (mm0OutDim ≠ mm2OutDim): structurally necessary.
        //   Example — moondream2: mm.0 [1152→8192] + GELU + mm.2 [8192→2048].
        //   Without mm.2 the patch vectors are 8192-dim but phi-2 needs 2048-dim.
        //   mm.2 is applied in project().
        //
        //   SQUARE (mm0OutDim == mm2OutDim): do NOT apply — 2026-07-12 regression.
        //   Example — LLaVA-1.5: applying the square mm.2 caused output to degenerate
        //   from "coherent but hallucinated" to a repeating <image>-token loop.
        //   Root cause unknown; skipping is confirmed correct for that model.
        //   mm.2 is loaded and logged for diagnostics but not applied in project().
        if (r.hasTensor("mm.2.weight")) {
            long[] proj2Dims = r.tensorDims("mm.2.weight");
            long mm2InDim  = proj2Dims[0];
            long mm2OutDim = proj2Dims[proj2Dims.length - 1];

            // mm.2 must chain from mm.0
            if (mm2InDim != this.mm0OutDim)
                throw new IllegalStateException("Vision projector mm.2.weight first dim " + mm2InDim
                        + " does not chain from mm.0 output dim " + this.mm0OutDim);

            this.finalOutDim = (int) mm2OutDim;
            projWeight2 = r.tensor("mm.2.weight");
            projBias2   = r.hasTensor("mm.2.bias") ? r.tensor("mm.2.bias") : null;
            if (projBias2 != null && projBias2.length != this.finalOutDim)
                throw new IllegalStateException("Vision projector mm.2.bias has length " + projBias2.length
                        + ", expected finalOutDim=" + this.finalOutDim);

            if (this.finalOutDim != this.mm0OutDim) {
                log.info("Vision projector: 2-layer MLP — mm.0 [" + H + "→" + mm0OutDim + "] + GELU"
                        + " + mm.2 [" + mm0OutDim + "→" + finalOutDim + "]  finalOutDim=" + finalOutDim
                        + " (non-square: mm.2 is structurally required and will be applied).");
            } else {
                log.info("Vision projector: mm.2.weight present but SQUARE [" + mm0OutDim + "→" + finalOutDim + "]"
                        + " — NOT applied (2026-07-12 regression for LLaVA-1.5; see javadoc/CHANGELOG).");
            }
        } else {
            this.finalOutDim = this.mm0OutDim;
            projWeight2 = null;
            projBias2   = null;
            log.info("Vision projector: mm.2.weight not found. Using mm.0 only (single linear layer).");
        }

        log.info("Vision encoder loaded — " + L + " blocks, hidden=" + H
                + " patches=" + cfg.numPatches() + " mm0OutDim=" + mm0OutDim + " finalOutDim=" + finalOutDim);
    }

    /**
     * Actual width of the vision projector's output — i.e. the dimension of
     * each patch vector returned by {@link #encode}.
     *
     * The dimension of each patch vector returned by {@link #encode} — i.e.
     * what gets spliced into the LLM's embedding space. This is the final
     * projector output:
     * <ul>
     *   <li>Single-layer projector (mm.0 only): mm.0's own output dim.</li>
     *   <li>Non-square 2-layer MLP (mm.0 expands, mm.2 contracts, shapes differ):
     *       mm.2's output dim. Example — moondream2: mm.0 [1152→8192] + GELU +
     *       mm.2 [8192→2048]; {@code outputDim()} returns 2048.</li>
     *   <li>Square mm.2 (LLaVA-1.5 pattern, not applied): mm.0's output dim.</li>
     * </ul>
     * Callers must use this method, not {@link VisionConfig#projectionDim()} —
     * that metadata field is unreliable across mmproj exports.
     */
    public int outputDim() {
        return finalOutDim;
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
        int H  = cfg.hiddenSize();
        int nP = cfg.numPatches();

        // Step 1 — patch embedding: linear projection of each raw patch
        float[][] tokens = patchEmbed(pixelTensor, H, nP);

        // Step 2 — build sequence (prepend CLS for CLIP; patches-only for SigLIP)
        float[][] seq = buildSequence(tokens, classEmbd);
        int N          = seq.length;   // nP+1 with CLS, nP without
        int patchStart = N - nP;       // 1 with CLS, 0 without

        // Step 3 — add position embeddings (v.position_embd.weight covers exactly N positions)
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < H; d++)
                seq[i][d] += posEmbd[i * H + d];
        }

        // Step 4 — pre-encoder layer norm
        for (int i = 0; i < N; i++)
            seq[i] = layerNorm(seq[i], preLnWeight, preLnBias, cfg.layerNormEps());

        // Step 5 — transformer blocks
        for (int li = 0; li < cfg.numLayers(); li++)
            seq = transformerBlock(seq, li, N, H);

        // Step 6 — vision projector on patch tokens only (CLS position skipped when present)
        float[][] out = new float[nP][];
        for (int i = 0; i < nP; i++)
            out[i] = project(seq[patchStart + i]);

        logPatchEmbeddingStats(out);

        return out;
    }

    /**
     * Diagnostic-only: logs min/max/mean/L2-norm across all projected patch
     * embeddings. Added 2026-07-13 to check whether patch embeddings have a
     * wildly different magnitude than real LLM token embeddings — a scale
     * mismatch here would make the transformer's residual stream effectively
     * ignore the image (falling back to language-model priors), which would
     * explain plausible-but-generic captions without any shape-level bug.
     * Compare these numbers against the equivalent stats logged for a real
     * text-token embedding in VisionAwareForwardPassHandler.
     */
    private void logPatchEmbeddingStats(float[][] patches) {
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        double sum = 0, sumSq = 0;
        long count = 0;
        double normSum = 0;
        float minNorm = Float.POSITIVE_INFINITY, maxNorm = Float.NEGATIVE_INFINITY;
        for (float[] patch : patches) {
            double normSq = 0;
            for (float v : patch) {
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
                sumSq += (double) v * v;
                normSq += (double) v * v;
                count++;
            }
            float norm = (float) Math.sqrt(normSq);
            normSum += norm;
            if (norm < minNorm) minNorm = norm;
            if (norm > maxNorm) maxNorm = norm;
        }
        double mean = sum / count;
        double std = Math.sqrt(sumSq / count - mean * mean);
        double meanNorm = normSum / patches.length;
        log.info(String.format(
                "Vision patch embeddings stats (numPatches=%d, dim=%d): min=%.4f max=%.4f mean=%.4f std=%.4f "
                        + "| per-patch L2 norm: min=%.4f mean=%.4f max=%.4f",
                patches.length, patches.length > 0 ? patches[0].length : 0, min, max, mean, std, minNorm, meanNorm,
                maxNorm));
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

    // ── MLP (activation per clip.use_gelu — see VisionConfig.useGelu) ──────

    private float[] mlp(float[] x, int li) {
        int I = cfg.intermediateSize();
        int H = cfg.hiddenSize();
        float[] hidden = backend.sgemv(ffnUp[li], x, I, H);
        for (int i = 0; i < I; i++)
            hidden[i] = activation(hidden[i] + bffnUp[li][i]);
        float[] out = backend.sgemv(ffnDown[li], hidden, H, I);
        for (int d = 0; d < H; d++)
            out[d] += bffnDown[li][d];
        return out;
    }

    /**
     * Dispatches to the activation this specific file was actually exported
     * with. {@code clip.use_gelu=true} → standard tanh-approx GELU.
     * {@code clip.use_gelu=false} → quick_gelu (OpenAI CLIP's original
     * activation, {@code x * sigmoid(1.702x)}).
     *
     * <p>2026-07-20: before this method existed, every call site here used
     * {@link #gelu} unconditionally, regardless of what the file declared.
     * Found via {@code ./juno gguf-info} that
     * llava-v1.5-7b-mmproj-Q4_0.gguf declares {@code clip.use_gelu=false} —
     * meaning every one of the 23 transformer blocks was silently using the
     * wrong activation function this entire session. Shape-valid, numerically
     * stable (both formulas are smooth and bounded), which is exactly why
     * this produced coherent-but-wrong output rather than an outright crash
     * or the degenerate collapse seen from the unrelated mm.2 regression.
     */
    private float activation(float x) {
        return cfg.useGelu() ? gelu(x) : quickGelu(x);
    }

    /** OpenAI CLIP's original activation: x * sigmoid(1.702x). */
    static float quickGelu(float x) {
        return x / (1f + (float) Math.exp(-1.702 * x));
    }

    // ── Vision projector: mm.0 -> [GELU -> mm.2] ───────────────────────────

    private float[] project(float[] x) {
        // Apply mm.2 only when it is structurally necessary — i.e. when mm.0's
        // output dim differs from the final output dim (moondream2: 1152→8192 then
        // 8192→2048). When mm.0 and mm.2 share the same width (LLaVA-1.5 square
        // pattern), NOT applying mm.2 is correct per the 2026-07-12 regression.
        boolean applyMm2 = (projWeight2 != null) && (finalOutDim != mm0OutDim);
        return applyProjector(backend, x,
                projWeight,  projBias,
                applyMm2 ? projWeight2 : null,
                applyMm2 ? projBias2   : null,
                cfg.hiddenSize(), mm0OutDim, finalOutDim);
    }

    /**
     * Builds the token sequence passed into the transformer encoder.
     *
     * <p>CLIP-style models provide a {@code v.class_embd} tensor and prepend a
     * CLS token; the returned sequence is {@code [CLS, patch_0, …, patch_{nP-1}]}
     * with length {@code nP+1}. SigLIP-style models (e.g. moondream2) have no CLS
     * token ({@code classEmbd == null}); the returned sequence is the patch array
     * itself, length {@code nP}.
     *
     * <p>Package-private and static for direct unit testing, consistent with
     * {@link #applyProjector}, {@link #resolveProjectorOutputDim}, and
     * {@link #resolveFfnOrientation}.
     *
     * @param patches    patch embeddings, shape [nP][H]; inner arrays shared in the
     *                   no-CLS path, copied into a new outer array in the CLS path
     * @param classEmbd  CLS token embedding [H], or {@code null} for SigLIP models
     * @return sequence array of length nP (no-CLS) or nP+1 (with-CLS)
     */
    static float[][] buildSequence(float[][] patches, float[] classEmbd) {
        if (classEmbd == null)
            return patches;
        int nP = patches.length;
        int H  = classEmbd.length;
        float[][] seq = new float[nP + 1][H];
        System.arraycopy(classEmbd, 0, seq[0], 0, H);
        for (int i = 0; i < nP; i++)
            seq[i + 1] = patches[i];
        return seq;
    }

    /**
     * Pure projector math (no field access): mm.0 linear, then — when
     * {@code w2} is non-null — GELU followed by mm.2 linear.
     *
     * <p>Two-layer behaviour ({@code w2 != null}):
     * <ul>
     *   <li>Non-square case ({@code mm0OutDim ≠ finalOutDim}): mm.0 expands
     *       {@code hiddenSize → mm0OutDim}, GELU, mm.2 contracts
     *       {@code mm0OutDim → finalOutDim}. Used by moondream2.</li>
     *   <li>Square case ({@code mm0OutDim == finalOutDim}): caller passes
     *       {@code w2 = null} to skip mm.2 (LLaVA-1.5 regression).</li>
     * </ul>
     *
     * Package-private and static so it is directly unit-testable without
     * constructing a full VisionEncoder from a synthetic GGUF file (same
     * pattern as {@link #resolveProjectorOutputDim}).
     *
     * @param backend     MatVec implementation to multiply with
     * @param x           input vector, length {@code hiddenSize}
     * @param w1          mm.0.weight, length {@code mm0OutDim * hiddenSize}
     * @param b1          mm.0.bias, length {@code mm0OutDim}, or {@code null}
     * @param w2          mm.2.weight, length {@code finalOutDim * mm0OutDim},
     *                    or {@code null} for single-layer / skipped mm.2
     * @param b2          mm.2.bias, length {@code finalOutDim}, or {@code null}
     * @param hiddenSize  vision encoder hidden size (mm.0's input width)
     * @param mm0OutDim   mm.0's output width (= mm.2's input width when w2 != null)
     * @param finalOutDim final output width: mm.2's output when applied,
     *                    else same as {@code mm0OutDim}
     */
    static float[] applyProjector(MatVec backend, float[] x, float[] w1, float[] b1, float[] w2, float[] b2,
            int hiddenSize, int mm0OutDim, int finalOutDim) {
        float[] out = backend.sgemv(w1, x, mm0OutDim, hiddenSize);
        if (b1 != null)
            for (int i = 0; i < mm0OutDim; i++)
                out[i] += b1[i];

        if (w2 == null)
            return out;

        for (int i = 0; i < mm0OutDim; i++)
            out[i] = gelu(out[i]);

        float[] out2 = backend.sgemv(w2, out, finalOutDim, mm0OutDim);
        if (b2 != null)
            for (int i = 0; i < finalOutDim; i++)
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