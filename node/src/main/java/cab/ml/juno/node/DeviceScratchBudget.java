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

/**
 * How much device memory has to stay free for inference once the weights are
 * uploaded.
 *
 * <p>Weight upload and the forward pass both allocate from the same device, but
 * only the upload was ever written to expect the device to run out. Uploading
 * layers until the allocator refuses therefore ends with a card that is full and
 * a model that loads, decodes single tokens, and then fails on the first prompt
 * wide enough to take the batched path, because by then nothing is left for the
 * prefill window.
 *
 * <p>What the batched path needs is window-shaped. Batched K-quant matmuls
 * multiply the packed weights directly, so the term is the prefill window
 * ({@link PrefillWindowFootprint}) at its narrowest capacity,
 * {@link #RESERVED_WINDOW_ROWS} rows; a wider window, which the adaptive prefill
 * chunk sizes from whatever is free after the upload, is not reserved for. Only
 * when the tiled kernel cannot load do batched K-quant matmuls dequantize a whole
 * weight matrix into an FP16 scratch first, and then that matrix is reserved for
 * as well ({@link #dequantScratchBytes}): for a 30B Llama, the FFN pair at
 * 6656 x 17920 halves, 227 MiB, against a 26 MiB window. Host staging and the
 * matmul library's workspace share the device and are covered by a proportional
 * margin rather than modelled individually; memory the allocator withholds is a
 * fixed allowance ({@link #ALLOCATOR_HOLDBACK_BYTES}).
 *
 * @author Yevhen Soldatov
 */
final class DeviceScratchBudget {

    /**
     * Prefill window rows kept free: the region's smallest window capacity, which
     * every window wider than the host path's {@link PrefillWindowRegion#MAX_HOST_WINDOW}
     * rows allocates at least.
     */
    static final int RESERVED_WINDOW_ROWS = PrefillWindowRegion.CAPACITY_STEP;

    /**
     * Extra fraction kept free on top of the window and any dequant scratch,
     * covering host-staged matmul buffers and the matmul library's workspace.
     */
    private static final long MARGIN_PERCENT = 40;

    /**
     * Device memory the free-memory query reports that no allocation can obtain:
     * with the device full, it still read 44 to 54 MiB free on a GTX 1080, a figure
     * that differs between processes and holds within one. The upload stop rule reads
     * that query, so this is kept free on top of everything else. A larger reserve hid
     * it; one sized to a prefill window does not.
     */
    static final long ALLOCATOR_HOLDBACK_BYTES = 64L * 1024 * 1024;

    /**
     * Context the GPU-attention KV mirror is reserved for on the layers whose weights
     * are on the device: a 512-token prompt keeps its attention on the device on a
     * card the model does not fit. A longer context grows the mirror past the
     * reserve, and running out then falls back to CPU attention, announced.
     */
    static final int RESERVED_MIRROR_TOKENS = 512;

    private DeviceScratchBudget() {
    }

    /**
     * Device bytes to keep free for one request's KV mirror: the layers whose weights
     * are on the device at {@link #RESERVED_MIRROR_TOKENS}, the rest at the
     * {@link DeviceKvCache#INITIAL_SEQ_CAPACITY} every layer is allocated at and only
     * device layers grow from, and the old buffer a layer holds while it grows (at most
     * half the reserved context).
     *
     * @param deviceLayers layers whose weights are on the device
     * @param hostLayers   the shard's other layers
     * @param kvDim        key/value width
     */
    static long kvMirrorReserveBytes(int deviceLayers, int hostLayers, int kvDim) {
        if (deviceLayers < 0)
            throw new IllegalArgumentException("deviceLayers must be >= 0 (got " + deviceLayers + ")");
        if (hostLayers < 0)
            throw new IllegalArgumentException("hostLayers must be >= 0 (got " + hostLayers + ")");
        long grown = kvMirrorBytes(deviceLayers, kvDim, RESERVED_MIRROR_TOKENS);
        long growing = kvMirrorBytes(Math.min(deviceLayers, 1), kvDim, RESERVED_MIRROR_TOKENS / 2);
        return grown + growing + kvMirrorBytes(hostLayers, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY);
    }

    /**
     * Device bytes of the FP16 scratch the dequantizing route expands one packed
     * weight matrix into: the widest matmul in the model.
     *
     * <p>Every matmul that can take the batched path is one of
     * {@code hidden x hidden}, {@code kvDim x hidden}, {@code ffn x hidden} or
     * {@code hidden x ffn}, so the widest is {@code hidden} times the largest of
     * the three dimensions.
     *
     * @throws IllegalArgumentException if any dimension is not positive, which
     *                                  would otherwise reserve nothing at all
     */
    static long dequantScratchBytes(int hiddenDim, int kvDim, int ffnDim) {
        if (hiddenDim < 1)
            throw new IllegalArgumentException("hiddenDim must be >= 1 (got " + hiddenDim + ")");
        if (kvDim < 1)
            throw new IllegalArgumentException("kvDim must be >= 1 (got " + kvDim + ")");
        if (ffnDim < 1)
            throw new IllegalArgumentException("ffnDim must be >= 1 (got " + ffnDim + ")");
        long widest = Math.max(hiddenDim, Math.max(kvDim, ffnDim));
        return (long) hiddenDim * widest * Short.BYTES;
    }

    /**
     * Device bytes to keep free for the forward pass: the reserved prefill window,
     * plus the dequant scratch when batched K-quant matmuls take the dequantizing
     * route, plus the margin, plus {@link #ALLOCATOR_HOLDBACK_BYTES}.
     *
     * @param windowBytes         the window at {@link #RESERVED_WINDOW_ROWS} rows
     * @param dequantScratchBytes {@link #dequantScratchBytes}, or 0 when no matmul dequantizes
     * @throws IllegalArgumentException if {@code windowBytes} is not positive or
     *                                  {@code dequantScratchBytes} is negative
     */
    static long reserveBytes(long windowBytes, long dequantScratchBytes) {
        if (windowBytes < 1)
            throw new IllegalArgumentException("windowBytes must be >= 1 (got " + windowBytes + ")");
        if (dequantScratchBytes < 0)
            throw new IllegalArgumentException("dequantScratchBytes must be >= 0 (got " + dequantScratchBytes + ")");
        long needed = windowBytes + dequantScratchBytes;
        return needed + needed * MARGIN_PERCENT / 100 + ALLOCATOR_HOLDBACK_BYTES;
    }

    /**
     * Device bytes the GPU-resident attention path needs for its KV mirror at
     * {@code initialTokens} of context: one K and one V buffer per layer.
     *
     * <p>This is the allocation that runs at the first token rather than at load,
     * which is why filling the card with weights defers the failure to the first
     * prompt instead of failing the load. {@link #kvMirrorReserveBytes} builds the
     * reserve from it: growth to {@link #RESERVED_MIRROR_TOKENS} is held free, growth
     * beyond it is a runtime concern, since reserving a long context up front would
     * pin memory that a short conversation never uses.
     *
     * @param layerCount   layers this handler owns, or 0 when no mirror is allocated
     * @param kvDim        key/value width
     * @param initialTokens context the mirror is first allocated for
     */
    static long kvMirrorBytes(int layerCount, int kvDim, int initialTokens) {
        if (layerCount <= 0)
            return 0L;
        if (kvDim < 1)
            throw new IllegalArgumentException("kvDim must be >= 1 (got " + kvDim + ")");
        if (initialTokens < 1)
            throw new IllegalArgumentException("initialTokens must be >= 1 (got " + initialTokens + ")");
        return 2L * layerCount * initialTokens * kvDim * Short.BYTES;
    }

    /**
     * Whether another weight layer can be uploaded while still leaving
     * {@code reserveBytes} free afterwards.
     *
     * <p>Both unknowns are treated as "do not block": a {@code freeBytes} of zero
     * means the free-memory query failed, and a {@code layerBytes} of zero means
     * no layer has been uploaded yet to measure. In either case this rule stands
     * aside and the pre-existing behaviour -- upload, and catch the allocator's
     * refusal -- still applies.
     *
     * @param freeBytes    device bytes currently free, or 0 if unknown
     * @param layerBytes   device bytes one layer costs, or 0 if not yet measured
     * @param reserveBytes bytes that must remain free, from {@link #reserveBytes}
     */
    static boolean canUploadAnotherLayer(long freeBytes, long layerBytes, long reserveBytes) {
        if (freeBytes <= 0 || layerBytes <= 0)
            return true;
        return freeBytes - layerBytes >= reserveBytes;
    }
}
