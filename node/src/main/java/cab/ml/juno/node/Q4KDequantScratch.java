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

import java.lang.foreign.MemorySegment;

/**
 * Per-thread device scratch holding one weight-shaped FP16 buffer: the dequant
 * output of whichever {@link DeviceQ4KMatrix} projection is mid-prefill.
 *
 * <p>Unlike {@code Fp16Scratch}'s {@code dXh}/{@code dY} (sized by batch), this
 * buffer is sized by {@code rows * cols} of the weight matrix being dequantized —
 * independent of batch. Grown lazily and kept at the largest size seen, same
 * grow-and-keep-max pattern as {@code Fp32Scratch}/{@code Fp16Scratch}; freed and
 * reallocated only when a larger shape is requested.
 */
final class Q4KDequantScratch {

    private MemorySegment dOutFp16;
    private long bytes;

    /** Returns a device FP16 buffer of at least {@code rows * cols} half-words, growing if needed. */
    MemorySegment ensure(CudaBindings cuda, int deviceIndex, long rows, long cols) {
        long needed = rows * cols * Short.BYTES;
        if (bytes < needed) {
            // Drop the old buffer from this object before freeing it, so a failed
            // allocation leaves an empty scratch rather than one pointing at memory
            // that has already been freed. Callers may survive the failure and come
            // back, and a stale pointer here surfaces later as an unrelated invalid
            // argument on a memcpy rather than as the allocation failure it was.
            MemorySegment previous = dOutFp16;
            dOutFp16 = null;
            bytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            dOutFp16 = cuda.deviceMalloc(deviceIndex, needed);
            bytes = needed;
        }
        return dOutFp16;
    }
}
