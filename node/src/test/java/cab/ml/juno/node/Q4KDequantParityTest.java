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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolates {@code Q4KMmqKernel.launchDequant}'s elementwise dequant-to-FP16
 * kernels from the batched-GEMM path built on top of them
 * ({@code CudaSgemmBatchedPrefillParityTest}): dequant a known Q4_K / Q5_K /
 * Q6_K row-major matrix on the device, download the FP16 result, and compare
 * against {@link GgufKQuantCodec#decodeRows} — the same reference decoder
 * used elsewhere for these quant types.
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=Q4KDequantParityTest}.
 */
@Tag("gpu")
@DisplayName("Q4KMmqKernel.launchDequant — parity vs GgufKQuantCodec.decodeRows")
class Q4KDequantParityTest {

    /** FP16 has ~3 decimal digits of precision; values here are in roughly [-4, 4]. */
    private static final float TOL = 5e-3f;

    private static GpuContext ctx;

    @BeforeAll
    static void init() {
        assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
        assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API — skipping");
        ctx = GpuContext.init(0);
        assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4K MMQ kernel failed to load");
    }

    @AfterAll
    static void destroy() {
        if (ctx != null)
            ctx.close();
    }

    @Test
    @DisplayName("Q4_K: device dequant matches GgufKQuantCodec.decodeRows")
    void q4k_dequant_matches_reference() {
        dequantMatchesReference(QuantizationLayout.TYPE_Q4_K, 3, 512, 4321);
    }

    @Test
    @DisplayName("Q5_K: device dequant matches GgufKQuantCodec.decodeRows")
    void q5k_dequant_matches_reference() {
        dequantMatchesReference(QuantizationLayout.TYPE_Q5_K, 3, 512, 8765);
    }

    @Test
    @DisplayName("Q6_K: device dequant matches GgufKQuantCodec.decodeRows")
    void q6k_dequant_matches_reference() {
        dequantMatchesReference(QuantizationLayout.TYPE_Q6_K, 3, 512, 1357);
    }

    private void dequantMatchesReference(int typeId, int rows, int cols, long seed) {
        Random rng = new Random(seed);
        float[] W = new float[rows * cols];
        for (int i = 0; i < W.length; i++)
            W[i] = (rng.nextFloat() * 2f) - 1f;

        byte[] raw = GgufKQuantCodec.encode(W, typeId);
        float[] expected = GgufKQuantCodec.decodeRows(raw, typeId, rows, cols);

        try (DeviceQ4KMatrix dW = DeviceQ4KMatrix.upload(ctx, raw, rows, cols, typeId)) {
            Q4KMmqKernel kernel = Q4KMmqKernel.tryLoad();
            CudaBindings cuda = CudaBindings.instance();
            long bytesOut = (long) rows * cols * Short.BYTES;
            MemorySegment dOut = cuda.deviceMalloc(ctx.deviceIndex(), bytesOut);
            try {
                // Launch on the default stream (null); the blocking cudaMemcpy below
                // synchronizes with it before reading the result.
                kernel.launchDequant(dW, dOut, null);

                try (Arena hostArena = Arena.ofConfined()) {
                    MemorySegment hostOut = hostArena.allocate(bytesOut);
                    CudaBindings.check(
                            CudaBindings.callInt(cuda.cudaMemcpy,
                                    hostOut, dOut, bytesOut, CudaBindings.D2H),
                            "cudaMemcpy(dequant D2H)");

                    float[] actual = new float[rows * cols];
                    for (int i = 0; i < actual.length; i++)
                        actual[i] = Float.float16ToFloat(hostOut.getAtIndex(JAVA_SHORT, i));

                    assertThat(actual).containsExactly(expected, within(TOL));
                }
            } finally {
                cuda.deviceFree(dOut);
            }
        }
    }
}
