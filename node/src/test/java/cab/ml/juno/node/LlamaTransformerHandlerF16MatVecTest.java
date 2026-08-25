package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Regression coverage for: any F16-weighted GGUF model (a very common,
 * standard format — not exotic) failed every inference request with
 * {@code UnsupportedOperationException: Quantized matVec not implemented for
 * GGML type 1 — add a case branch or convert to float[] first.}
 *
 * Root cause: {@code LlamaTransformerHandler}'s two dispatches over
 * {@code GgufReader.QuantizedTensor.type()} — {@code matVecQuantizedNoEvent}
 * (CPU path) and {@code dequantize} (CUDA upload path) — handled GGML types
 * 0 (F32), 8 (Q8_0), and 10–14 (Q2_K..Q6_K), but never 1 (F16), even though
 * F16 is one of the most common GGUF weight formats. Any model file
 * literally named {@code *-f16.gguf} hit this immediately on its first
 * forward pass.
 *
 * Fix: added a case 1 branch to both switches, backed by
 * {@code matVecF16raw}/{@code dequantizeF16}, reusing the same
 * {@link GgufReader#f16ToF32} conversion {@code GgufReader.loadF16} already
 * uses, so results are bit-identical to eager float[] dequantization.
 */
@DisplayName("LlamaTransformerHandler — GGML type 1 (F16) quantized matVec/dequantize")
class LlamaTransformerHandlerF16MatVecTest {

    /** Encodes a float as its nearest IEEE-754 half-precision bit pattern. */
    private static short toHalfBits(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;
        if (exp <= 0) return (short) sign;              // flushes to zero — fine for these test values
        if (exp >= 31) return (short) (sign | 0x7C00);   // overflow → inf
        return (short) (sign | (exp << 10) | (mantissa >>> 13));
    }

    /** Builds an F16-typed (type=1) QuantizedTensor from plain float values. */
    private static GgufReader.QuantizedTensor f16Tensor(float[] values) {
        ByteBuffer bb = ByteBuffer.allocate(values.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) bb.putShort(toHalfBits(v));
        return new GgufReader.QuantizedTensor("test.weight", 1, values.length, bb.array());
    }

    @Test
    @DisplayName("matVec on an F16 tensor no longer throws, and is numerically correct")
    void f16_matVec_is_supported_and_correct() {
        // A = [[1,2],[3,4]] (F16), x = [1,0] → y = [1,3]
        GgufReader.QuantizedTensor A = f16Tensor(new float[] { 1f, 2f, 3f, 4f });

        float[] y = LlamaTransformerHandler.matVec(A, new float[] { 1f, 0f }, 2, 2);
        assertThat(y[0]).isCloseTo(1f, offset(1e-3f));
        assertThat(y[1]).isCloseTo(3f, offset(1e-3f));

        float[] y2 = LlamaTransformerHandler.matVec(A, new float[] { 0f, 1f }, 2, 2);
        assertThat(y2[0]).isCloseTo(2f, offset(1e-3f));
        assertThat(y2[1]).isCloseTo(4f, offset(1e-3f));
    }

    @Test
    @DisplayName("matVec on an F16 tensor matches a plain-float reference dot product (larger, non-trivial case)")
    void f16_matVec_matches_reference_dot_product() {
        int rows = 3;
        int cols = 5;
        float[] weights = { 0.5f, -1.25f, 2f, 0.1f, -0.75f,
                             1f, 1f, 1f, 1f, 1f,
                             -2f, 3.5f, 0.25f, -0.125f, 4f };
        float[] x = { 1f, 2f, -1f, 0.5f, 3f };

        GgufReader.QuantizedTensor A = f16Tensor(weights);
        float[] actual = LlamaTransformerHandler.matVec(A, x, rows, cols);

        for (int r = 0; r < rows; r++) {
            float expected = 0f;
            for (int c = 0; c < cols; c++) expected += weights[r * cols + c] * x[c];
            assertThat(actual[r]).as("row " + r).isCloseTo(expected, offset(1e-2f));
        }
    }

    @Test
    @DisplayName("dequantize() on an F16 tensor no longer throws, and is numerically correct")
    void f16_dequantize_is_supported_and_correct() {
        float[] values = { 1f, -2f, 0.5f, 4f };
        GgufReader.QuantizedTensor A = f16Tensor(values);

        float[] out = LlamaTransformerHandler.dequantize(A, 2, 2);

        for (int i = 0; i < values.length; i++)
            assertThat(out[i]).isCloseTo(values[i], offset(1e-3f));
    }
}