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
 * Device buffer for a prefill window's activations packed as Q8_1, the input of
 * {@link KQuantGemmKernel}: 36 bytes per 32 values, so about half the window's
 * FP16 size, against the whole FP16 weight matrix the dequantizing route kept
 * ({@link Q4KDequantScratch}). Grown on demand and kept at the largest size;
 * same ownership and locking as the other {@link CudaMatVec} scratch.
 */
final class Q8WindowScratch {

    private MemorySegment dQ8;
    private long bytes;

    /** Device bytes held now. */
    long heldBytes() {
        return bytes;
    }

    /** Frees the buffer; the next {@link #ensure} allocates again. */
    void release(CudaBindings cuda) {
        MemorySegment previous = dQ8;
        dQ8 = null;
        bytes = 0L;
        if (previous != null)
            cuda.deviceFree(previous);
    }

    /** Returns a device buffer of at least {@code needed} bytes, growing if needed. */
    MemorySegment ensure(CudaBindings cuda, int deviceIndex, long needed) {
        if (bytes < needed) {
            // As Q4KDequantScratch: drop the old buffer before freeing it, so a failed
            // allocation leaves an empty scratch rather than a stale pointer.
            MemorySegment previous = dQ8;
            dQ8 = null;
            bytes = 0L;
            if (previous != null)
                cuda.deviceFree(previous);
            dQ8 = cuda.deviceMalloc(deviceIndex, needed);
            bytes = needed;
        }
        return dQ8.reinterpret(bytes);
    }
}
