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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that {@link CpuMatVec#sgemm} produces per-row results identical to
 * calling {@link CpuMatVec#sgemv} B times — the primary business-logic
 * correctness requirement for batched prefill.
 *
 * <p>All cases use exact float equality because {@link CpuMatVec#sgemm} uses
 * the same inner-loop reduction order as {@link CpuMatVec#sgemv} (row-major,
 * sequential inner loop) — bitwise parity is achievable on the CPU path.
 */
class CpuMatVecSgemmTest {

    private CpuMatVec backend;

    @BeforeEach
    void setUp() {
        backend = new CpuMatVec();
    }

    @Test
    void sgemmMatchesSgemvForEveryBatchSlot() {
        int rows = 16;
        int cols = 32;
        int B = 8;

        Random rng = new Random(0xBEEF_CAFE);
        float[] A = randomFloats(rng, rows * cols);
        float[][] X = new float[B][];
        for (int b = 0; b < B; b++) X[b] = randomFloats(rng, cols);

        float[][] Y = backend.sgemm(A, X, rows, cols);

        assertEquals(B, Y.length);
        for (int b = 0; b < B; b++) {
            float[] expected = backend.sgemv(A, X[b], rows, cols);
            assertArrayEquals(expected, Y[b], 0f,
                    "sgemm row " + b + " must match sgemv output exactly (same float reduction order)");
        }
    }

    @Test
    void sgemmSingleBatchIsEquivalentToSgemv() {
        int rows = 64;
        int cols = 128;

        Random rng = new Random(0xCAFE_1234);
        float[] A = randomFloats(rng, rows * cols);
        float[] x = randomFloats(rng, cols);

        float[][] Y = backend.sgemm(A, new float[][]{ x }, rows, cols);

        assertArrayEquals(backend.sgemv(A, x, rows, cols), Y[0], 0f,
                "single-batch sgemm must equal sgemv exactly");
    }

    @Test
    void sgemmZeroInputProducesZeroOutput() {
        int rows = 8;
        int cols = 16;
        int B = 4;

        float[] A = randomFloats(new Random(42), rows * cols);
        float[][] X = new float[B][cols]; // all zeros

        float[][] Y = backend.sgemm(A, X, rows, cols);

        for (int b = 0; b < B; b++) {
            for (float v : Y[b]) assertEquals(0f, v, "A * 0 must be 0");
        }
    }

    @Test
    void sgemmRejectsWrongALength() {
        assertThrows(IllegalArgumentException.class, () ->
            backend.sgemm(new float[10], new float[][]{ new float[8] }, 4, 8),
            "A.length mismatch must throw"
        );
    }

    @Test
    void sgemmRejectsWrongBatchVectorLength() {
        assertThrows(IllegalArgumentException.class, () ->
            backend.sgemm(new float[32], new float[][]{ new float[3] }, 4, 8),
            "X[b].length != cols must throw"
        );
    }

    @Test
    void sgemmEmptyBatchReturnsEmptyResult() {
        float[] A = new float[16]; // 4x4
        float[][] result = backend.sgemm(A, new float[0][], 4, 4);
        assertEquals(0, result.length);
    }

    @Test
    void sgemmLargeBatchDimensionsForPerfBound() {
        // Typical vision prefill window: 576 patch tokens, hiddenDim=4096 → 256 cols here
        int rows = 64;
        int cols = 256;
        int B = 576;

        Random rng = new Random(99);
        float[] A = randomFloats(rng, rows * cols);
        float[][] X = new float[B][];
        for (int b = 0; b < B; b++) X[b] = randomFloats(rng, cols);

        float[][] Y = backend.sgemm(A, X, rows, cols);
        assertEquals(B, Y.length);

        // Sample-check 8 slots to keep test fast
        for (int b = 0; b < B; b += B / 8) {
            assertArrayEquals(backend.sgemv(A, X[b], rows, cols), Y[b], 0f,
                    "spot-check slot " + b);
        }
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextFloat() * 2f - 1f;
        return a;
    }
}
