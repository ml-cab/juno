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
import java.util.logging.Logger;

/**
 * Per-shard GPU weight store: every projection matrix dequantized once at load
 * time and pinned on the device as a {@link DeviceFloatMatrix}.
 *
 * <p>This eliminates the H2D copy of A on every {@code cublasSgemv} call. Without
 * it, {@link CudaMatVec#sgemv(float[], float[], int, int)} reallocates device
 * memory and re-transfers the full weight matrix on every token step — saturating
 * PCIe bandwidth and leaving GPU-Util at 0%.
 *
 * <p>Memory per shard (TinyLlama, float32):
 * <ul>
 *   <li>wq/wo: 2048x2048x4 = 16 MB/layer
 *   <li>wk/wv: 256x2048x4  =  2 MB/layer
 *   <li>wGate/wUp/wDown: 5632x2048x4 = 46 MB/layer
 * </ul>
 * A 7-layer shard uses ~1.2 GB VRAM. On machines where multiple JVM nodes share
 * a single small GPU (e.g. MX150 2 GB), {@link #upload} returns {@code null}
 * instead of crashing, and the caller falls back to the CPU matVec path.
 *
 * <p>Lifecycle: create once in
 * {@link LlamaTransformerHandler#load(java.nio.file.Path, ShardContext, MatVec)},
 * release with {@link #close()} when the shard is unloaded.
 *
 * @author Yevhen Soldatov
 */
public final class GpuWeightShard implements AutoCloseable {

    private static final Logger log = Logger.getLogger(GpuWeightShard.class.getName());

    private final DeviceFloatMatrix[] wq;
    private final DeviceFloatMatrix[] wk;
    private final DeviceFloatMatrix[] wv;
    private final DeviceFloatMatrix[] wo;
    private final DeviceFloatMatrix[] wGate;
    private final DeviceFloatMatrix[] wUp;
    private final DeviceFloatMatrix[] wDown;
    /** Null when this shard is not the last node (hasOutputProjection == false). */
    private final DeviceFloatMatrix outputProj;

    private volatile boolean closed;

    private GpuWeightShard(
            DeviceFloatMatrix[] wq,
            DeviceFloatMatrix[] wk,
            DeviceFloatMatrix[] wv,
            DeviceFloatMatrix[] wo,
            DeviceFloatMatrix[] wGate,
            DeviceFloatMatrix[] wUp,
            DeviceFloatMatrix[] wDown,
            DeviceFloatMatrix outputProj) {
        this.wq         = wq;
        this.wk         = wk;
        this.wv         = wv;
        this.wo         = wo;
        this.wGate      = wGate;
        this.wUp        = wUp;
        this.wDown      = wDown;
        this.outputProj = outputProj;
    }

    /**
     * Dequantize every projection tensor for layers {@code [startLayer, endLayer)}
     * and upload each to device memory exactly once.
     *
     * <p>Returns {@code null} when VRAM is insufficient (cudaErrorMemoryAllocation).
     * Any partially-uploaded matrices are freed before returning so device memory
     * is left clean. This happens naturally when multiple JVM nodes share a single
     * small GPU: each node tries to upload concurrently and one or more will OOM.
     * The caller treats null as a signal to stay on the CPU matVec path.
     *
     * @param r             open GGUF reader for the model file
     * @param cfg           model config (dimension metadata)
     * @param startLayer    inclusive global layer index
     * @param endLayer      exclusive global layer index
     * @param hasOutputProj true for the last node shard (carries the lm_head)
     * @param ctx           open GPU context
     * @return fully-populated shard, or {@code null} if VRAM was exhausted
     */
    public static GpuWeightShard upload(
            GgufReader r,
            LlamaConfig cfg,
            int startLayer,
            int endLayer,
            boolean hasOutputProj,
            GpuContext ctx) throws IOException {

        int L   = endLayer - startLayer;
        int H   = cfg.hiddenDim();
        int kvD = cfg.kvDim();
        int I   = cfg.intermediateSize();

        DeviceFloatMatrix[] wq    = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wk    = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wv    = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wo    = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wGate = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wUp   = new DeviceFloatMatrix[L];
        DeviceFloatMatrix[] wDown = new DeviceFloatMatrix[L];
        DeviceFloatMatrix   outputProjDev = null;

        long uploadedBytes = 0;

        try {
            for (int li = 0; li < L; li++) {
                int i = li + startLayer;

                wq[li]    = upload(ctx, r, "blk." + i + ".attn_q.weight",      H,   H);
                wk[li]    = upload(ctx, r, "blk." + i + ".attn_k.weight",      kvD, H);
                wv[li]    = upload(ctx, r, "blk." + i + ".attn_v.weight",      kvD, H);
                wo[li]    = upload(ctx, r, "blk." + i + ".attn_output.weight", H,   H);
                wGate[li] = upload(ctx, r, "blk." + i + ".ffn_gate.weight",    I,   H);
                wUp[li]   = upload(ctx, r, "blk." + i + ".ffn_up.weight",      I,   H);
                wDown[li] = upload(ctx, r, "blk." + i + ".ffn_down.weight",    H,   I);

                uploadedBytes += bytesOf(wq[li], wk[li], wv[li], wo[li], wGate[li], wUp[li], wDown[li]);
            }

            if (hasOutputProj) {
                String outName = r.hasTensor("output.weight") ? "output.weight" : "token_embd.weight";
                outputProjDev  = upload(ctx, r, outName, cfg.vocabSize(), H);
                uploadedBytes += (long) cfg.vocabSize() * H * 4;
            }

        } catch (IllegalStateException e) {
            if (e.getMessage() != null && e.getMessage().startsWith("cudaMalloc failed")) {
                // VRAM exhausted — free whatever was partially allocated and fall back.
                closeAll(wq); closeAll(wk); closeAll(wv);  closeAll(wo);
                closeAll(wGate); closeAll(wUp); closeAll(wDown);
                if (outputProjDev != null) outputProjDev.close();
                log.warning(String.format(
                    "GpuWeightShard — VRAM exhausted after %.1f MB (layers %d-%d);"
                    + " falling back to CPU matVec for this shard",
                    uploadedBytes / 1_048_576.0, startLayer, endLayer));
                return null;
            }
            throw e;
        }

        log.info(String.format(
            "GpuWeightShard — %d layers uploaded to GPU device %d  (%.1f MB float32)",
            L, ctx.deviceIndex(), uploadedBytes / 1_048_576.0));

        return new GpuWeightShard(wq, wk, wv, wo, wGate, wUp, wDown, outputProjDev);
    }

    // ── Accessors (by local layer index li in [0, L)) ─────────────────────────

    public DeviceFloatMatrix wq(int li)    { return wq[li]; }
    public DeviceFloatMatrix wk(int li)    { return wk[li]; }
    public DeviceFloatMatrix wv(int li)    { return wv[li]; }
    public DeviceFloatMatrix wo(int li)    { return wo[li]; }
    public DeviceFloatMatrix wGate(int li) { return wGate[li]; }
    public DeviceFloatMatrix wUp(int li)   { return wUp[li]; }
    public DeviceFloatMatrix wDown(int li) { return wDown[li]; }

    /**
     * Output projection (lm_head). Only valid when this shard was created with
     * {@code hasOutputProj == true}.
     *
     * @throws IllegalStateException if this is not the last-node shard
     */
    public DeviceFloatMatrix outputProj() {
        if (outputProj == null)
            throw new IllegalStateException("outputProj not loaded — not the last shard");
        return outputProj;
    }

    /** Free all device memory held by this shard. Safe to call more than once. */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        closeAll(wq);
        closeAll(wk);
        closeAll(wv);
        closeAll(wo);
        closeAll(wGate);
        closeAll(wUp);
        closeAll(wDown);
        if (outputProj != null) outputProj.close();
        log.info("GpuWeightShard closed — device memory freed");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static DeviceFloatMatrix upload(GpuContext ctx, GgufReader r, String name,
                                             int rows, int cols) throws IOException {
        float[] host = r.tensor(name);
        return DeviceFloatMatrix.upload(ctx, host, rows, cols);
    }

    private static void closeAll(DeviceFloatMatrix[] arr) {
        if (arr == null) return;
        for (DeviceFloatMatrix m : arr) {
            if (m != null) m.close();
        }
    }

    private static long bytesOf(DeviceFloatMatrix... mats) {
        long sum = 0;
        for (DeviceFloatMatrix m : mats)
            sum += (long) m.rows() * m.cols() * 4;
        return sum;
    }
}