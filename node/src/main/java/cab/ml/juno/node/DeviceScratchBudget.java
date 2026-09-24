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
 * wide enough to take the batched path -- because that path dequantizes a whole
 * packed weight matrix into a device scratch buffer, and by then nothing is
 * left to put it in.
 *
 * <p>The dominant term is that scratch buffer. It holds one dequantized matrix
 * at a time, so the figure to reserve is the widest matmul in the model: for a
 * 30B Llama, the FFN pair at 6656 x 17920 halves, which is 227 MiB. Staging
 * buffers and per-request activations share the same device and are much
 * smaller, so they are covered by a proportional margin rather than modelled
 * individually.
 *
 * @author Yevhen Soldatov
 */
final class DeviceScratchBudget {

    /**
     * Extra fraction of the largest matrix kept free on top of it, covering the
     * FP16 staging buffers and per-request activations that share the device.
     */
    private static final long MARGIN_PERCENT = 40;

    private DeviceScratchBudget() {
    }

    /**
     * Device bytes to keep free for the forward pass, for a model of these
     * dimensions.
     *
     * <p>Every matmul that can take the batched dequant path is one of
     * {@code hidden x hidden}, {@code kvDim x hidden}, {@code ffn x hidden} or
     * {@code hidden x ffn}, so the widest is {@code hidden} times the largest of
     * the three dimensions.
     *
     * @throws IllegalArgumentException if any dimension is not positive, which
     *                                  would otherwise reserve nothing at all
     */
    static long reserveBytes(int hiddenDim, int kvDim, int ffnDim) {
        if (hiddenDim < 1)
            throw new IllegalArgumentException("hiddenDim must be >= 1 (got " + hiddenDim + ")");
        if (kvDim < 1)
            throw new IllegalArgumentException("kvDim must be >= 1 (got " + kvDim + ")");
        if (ffnDim < 1)
            throw new IllegalArgumentException("ffnDim must be >= 1 (got " + ffnDim + ")");
        long widest = Math.max(hiddenDim, Math.max(kvDim, ffnDim));
        long largestMatrixBytes = (long) hiddenDim * widest * Short.BYTES;
        return largestMatrixBytes + largestMatrixBytes * MARGIN_PERCENT / 100;
    }

    /**
     * Device bytes the GPU-resident attention path needs for its KV mirror at
     * {@code initialTokens} of context: one K and one V buffer per layer.
     *
     * <p>This is the allocation that runs at the first token rather than at load,
     * which is why filling the card with weights defers the failure to the first
     * prompt instead of failing the load. Only the initial capacity is reserved.
     * The mirror grows as a conversation lengthens, and that growth is a runtime
     * concern: it cannot be reserved for up front without pinning gigabytes that
     * a short conversation would never use.
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
