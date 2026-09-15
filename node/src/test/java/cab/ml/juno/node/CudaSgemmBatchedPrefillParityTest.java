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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Parity oracle for {@code CudaMatVec.sgemm(DeviceHalfMatrix|DeviceQ4KMatrix, float[][])}
 * across the small-batch (strided GEMV) / large-batch (tiled GEMM) threshold
 * ({@code HALF_SGEMM_BATCH_MAX = 8}): batched output must equal the per-{@code b}
 * serial {@code sgemv} loop, for every batch size on both sides of the cutover.
 *
 * <p>The {@link DeviceQ4KMatrix} cases currently exercise the correctness-preserving
 * serial default ({@code MatVec#sgemm(DeviceQ4KMatrix, float[][])}) for every batch —
 * no dequant-to-FP16 tiled-GEMM override exists yet. They stay green unmodified once
 * that override lands, since the oracle (serial per-row {@code sgemv}) does not change.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=CudaSgemmBatchedPrefillParityTest}.
 */
@Tag("gpu")
@DisplayName("CudaMatVec batched-prefill sgemm — parity vs serial sgemv loop")
class CudaSgemmBatchedPrefillParityTest {

    private static final float TOL = 5e-2f;

    private static GpuContext ctx;
    private static CudaMatVec mv;

    @BeforeAll
    static void init() {
        assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
        ctx = GpuContext.init(0);
        mv = new CudaMatVec(ctx);
    }

    @AfterAll
    static void destroy() {
        if (ctx != null)
            ctx.close();
    }

    private static final int[] BATCHES = { 1, 8, 9, 16, 32, 128 };

    @Test
    @DisplayName("DeviceHalfMatrix: sgemm matches serial sgemv loop for every batch across the threshold")
    void half_batched_matches_serial() {
        Random rng = new Random(1234);
        int rows = 17, cols = 13; // non-tile-aligned shape
        float[] W = random(rng, rows * cols);

        try (DeviceHalfMatrix dW = mv.uploadHalf(W, rows, cols)) {
            for (int batch : BATCHES) {
                float[][] X = new float[batch][];
                for (int b = 0; b < batch; b++)
                    X[b] = random(rng, cols);

                float[][] expected = new float[batch][];
                for (int b = 0; b < batch; b++)
                    expected[b] = mv.sgemv(dW, X[b]);

                float[][] actual = mv.sgemm(dW, X);
                assertThat(actual.length).as("batch=" + batch).isEqualTo(batch);
                for (int b = 0; b < batch; b++)
                    assertThat(actual[b]).as("batch=" + batch + " row=" + b)
                            .containsExactly(expected[b], within(TOL));
            }
        }
    }

    @Test
    @DisplayName("DeviceQ4KMatrix: sgemm matches serial sgemv loop for every batch across the threshold")
    void q4k_batched_matches_serial() {
        assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
        assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4K MMQ kernel failed to load");

        Random rng = new Random(5678);
        int rows = 13, cols = 256; // cols must be a Q4_K super-block multiple
        float[] W = random(rng, rows * cols);
        byte[] raw = GgufKQuantCodec.encode(W, QuantizationLayout.TYPE_Q4_K);

        try (DeviceQ4KMatrix dW = DeviceQ4KMatrix.upload(ctx, raw, rows, cols)) {
            for (int batch : BATCHES) {
                float[][] X = new float[batch][];
                for (int b = 0; b < batch; b++)
                    X[b] = random(rng, cols);

                float[][] expected = new float[batch][];
                for (int b = 0; b < batch; b++)
                    expected[b] = mv.sgemv(dW, X[b]);

                float[][] actual = mv.sgemm(dW, X);
                assertThat(actual.length).as("batch=" + batch).isEqualTo(batch);
                for (int b = 0; b < batch; b++)
                    assertThat(actual[b]).as("batch=" + batch + " row=" + b)
                            .containsExactly(expected[b], within(TOL));
            }
        }
    }

    private static float[] random(Random rng, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (rng.nextFloat() * 2f) - 1f;
        return v;
    }
}
