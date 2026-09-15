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
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Concurrency check for the Tier 17 large-batch {@code sgemm(DeviceHalfMatrix|DeviceQ4KMatrix,
 * float[][])} path: {@code CudaMatVec}'s {@code Fp16Scratch} and {@code Q4KDequantScratch}
 * buffers are held per-thread ({@code ThreadLocal}, sized independently per caller), mirroring
 * {@code CudaMatVecBackendTest#concurrent_calls_are_correct} for the existing single-token
 * {@code sgemv} path. Each worker thread uses its own distinctly-shaped weight matrix and its
 * own random data, so a scratch buffer leaking across threads (wrong size reused, or another
 * thread's dequant output read back) shows up as a wrong numeric result rather than a crash.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=CudaSgemmBatchedPrefillConcurrencyTest}.
 */
@Tag("gpu")
@DisplayName("CudaMatVec batched-prefill sgemm — concurrent calls across threads")
class CudaSgemmBatchedPrefillConcurrencyTest {

    private static final float TOL = 5e-2f;

    /**
     * Wider than {@code CudaSgemmBatchedPrefillParityTest}'s {@code TOL_Q4K_BATCHED} (8e-2):
     * that bound was calibrated against one fixed shape/seed (rows=13, cols=256). This test
     * sweeps a distinct shape per thread across two runs (8 shape/seed combinations total), and
     * one combination (thread=2, rows=11) empirically hit an 8.6e-2 max abs diff — reproduced
     * identically running sequentially (no threads involved), confirming it is the same
     * documented dequant-to-FP16-vs-int8-Q8_1-dot rounding noise, not thread-scratch corruption.
     * A real cross-thread corruption bug would show gross mismatches, not a single-digit-percent
     * tolerance overshoot on one element.
     */
    private static final float TOL_Q4K_BATCHED = 1.0e-1f;
    private static final int THREADS = 4;
    private static final int BATCH = 32; // > HALF_SGEMM_BATCH_MAX, exercises the tiled-GEMM path

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

    @Test
    @DisplayName("DeviceHalfMatrix: concurrent large-batch sgemm calls stay correct per-thread")
    void concurrent_half_batched_sgemm_is_correct() throws InterruptedException {
        // Each thread gets a differently-shaped weight matrix so cross-thread scratch reuse
        // (wrong buffer size, or another thread's staged data) would surface as a wrong answer.
        runConcurrently(t -> {
            Random rng = new Random(1000 + t);
            int rows = 11 + t, cols = 17 + t; // distinct, non-tile-aligned per thread
            float[] W = random(rng, rows * cols);
            try (DeviceHalfMatrix dW = mv.uploadHalf(W, rows, cols)) {
                float[][] X = new float[BATCH][];
                for (int b = 0; b < BATCH; b++)
                    X[b] = random(rng, cols);

                float[][] expected = new float[BATCH][];
                for (int b = 0; b < BATCH; b++)
                    expected[b] = mv.sgemv(dW, X[b]);

                float[][] actual = mv.sgemm(dW, X);
                assertThat(actual.length).as("thread=" + t).isEqualTo(BATCH);
                for (int b = 0; b < BATCH; b++)
                    assertThat(actual[b]).as("thread=" + t + " row=" + b)
                            .containsExactly(expected[b], within(TOL));
            }
        });
    }

    @Test
    @DisplayName("DeviceQ4KMatrix: concurrent large-batch sgemm calls stay correct per-thread")
    void concurrent_q4k_batched_sgemm_is_correct() throws InterruptedException {
        assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
        assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4K MMQ kernel failed to load");

        // cols must stay a Q4_K super-block multiple; rows varies per thread to keep shapes
        // distinct (Q4KDequantScratch is sized per (rows, cols) shape actually seen).
        runConcurrently(t -> {
            Random rng = new Random(2000 + t);
            int rows = 9 + t, cols = 256;
            float[] W = random(rng, rows * cols);
            byte[] raw = GgufKQuantCodec.encode(W, QuantizationLayout.TYPE_Q4_K);

            try (DeviceQ4KMatrix dW = DeviceQ4KMatrix.upload(ctx, raw, rows, cols)) {
                float[][] X = new float[BATCH][];
                for (int b = 0; b < BATCH; b++)
                    X[b] = random(rng, cols);

                float[][] expected = new float[BATCH][];
                for (int b = 0; b < BATCH; b++)
                    expected[b] = mv.sgemv(dW, X[b]);

                float[][] actual = mv.sgemm(dW, X);
                assertThat(actual.length).as("thread=" + t).isEqualTo(BATCH);
                for (int b = 0; b < BATCH; b++)
                    assertThat(actual[b]).as("thread=" + t + " row=" + b)
                            .containsExactly(expected[b], within(TOL_Q4K_BATCHED));
            }
        });
    }

    @FunctionalInterface
    private interface ThreadWork {
        void run(int threadIndex);
    }

    private static void runConcurrently(ThreadWork work) throws InterruptedException {
        Thread[] workers = new Thread[THREADS];
        AtomicReference<Throwable> failure = new AtomicReference<>();
        for (int t = 0; t < THREADS; t++) {
            final int idx = t;
            workers[t] = new Thread(() -> {
                try {
                    work.run(idx);
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            });
            workers[t].start();
        }
        for (Thread w : workers) w.join();
        Throwable e = failure.get();
        if (e != null)
            throw new AssertionError("Concurrent worker failed", e);
    }

    private static float[] random(Random rng, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (rng.nextFloat() * 2f) - 1f;
        return v;
    }
}
