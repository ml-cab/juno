package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import jdk.jfr.Recording;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies that the quantized {@code matVec} overloads in
 * {@link LlamaTransformerHandler} emit JFR events with {@code backend = "cpu"}.
 *
 * <p>Bug 2 root cause: the old {@code matVecQuantBackendLabel()} returned
 * strings like {@code "quantized-q4_k"}, so {@link jdk.jfr.consumer.RecordedEvent}s
 * were bucketed under unknown labels and {@code juno.MatVec.backend.cpu.count}
 * was always 0 — even though all quantized ops run purely on CPU via
 * {@link java.util.stream.IntStream#parallel()}.
 *
 * <p>The fix labels all quantized matVec events as {@code "cpu"}, matching
 * {@link CpuMatVec#sgemv} and enabling the JFR metrics extractor to count them
 * correctly under {@code juno.MatVec.backend.cpu.*}.
 */
@DisplayName("LlamaTransformerHandler — quantized matVec JFR backend label = \"cpu\"")
class MatVecQuantizedBackendLabelTest {

    private static final String MAT_VEC_EVENT = "juno.MatVec";

    @TempDir
    Path tmp;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Build a minimal F32-typed (type=0) QuantizedTensor backed by the given
     * float array. F32 is the simplest quantization format to construct in tests;
     * the backend-label behaviour under test is identical for all quant types.
     */
    private static GgufReader.QuantizedTensor f32Tensor(float[] data) {
        ByteBuffer bb = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) bb.putFloat(v);
        return new GgufReader.QuantizedTensor("test.weight", 0, data.length, bb.array());
    }

    /** Trivial 2×2 F32 weight matrix: [[1,2],[3,4]]. */
    private static GgufReader.QuantizedTensor twoByTwoMatrix() {
        return f32Tensor(new float[]{1f, 2f, 3f, 4f});
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("matVec(QuantizedTensor) emits JFR event with backend=\"cpu\"")
    void quantized_matVec_emits_cpu_backend_label(@TempDir Path jfrDir) throws Exception {
        Path jfrFile = jfrDir.resolve("matVec_label.jfr");

        try (Recording rec = new Recording()) {
            rec.enable(MAT_VEC_EVENT);
            rec.start();

            // Exercise the quantized overload — 2 rows, 2 cols, F32 tensor
            GgufReader.QuantizedTensor A = twoByTwoMatrix();
            float[] x = {1f, 0f};
            LlamaTransformerHandler.matVec(A, x, 2, 2);

            rec.stop();
            rec.dump(jfrFile);
        }

        List<RecordedEvent> matVecEvents = new ArrayList<>();
        try (RecordingFile rf = new RecordingFile(jfrFile)) {
            while (rf.hasMoreEvents()) {
                RecordedEvent ev = rf.readEvent();
                if (MAT_VEC_EVENT.equals(ev.getEventType().getName())) {
                    matVecEvents.add(ev);
                }
            }
        }

        assertThat(matVecEvents)
                .as("Expected at least one juno.MatVec JFR event")
                .isNotEmpty();

        for (RecordedEvent ev : matVecEvents) {
            String backend = ev.getString("backend");
            assertThat(backend)
                    .as("Quantized matVec must emit backend=\"cpu\" so juno.MatVec.backend.cpu.count "
                            + "is populated — got: " + backend)
                    .isEqualTo("cpu");
        }
    }

    @Test
    @DisplayName("matVec(QuantizedTensor) backend label is not a quantization-type string")
    void quantized_matVec_does_not_emit_quant_type_label(@TempDir Path jfrDir) throws Exception {
        Path jfrFile = jfrDir.resolve("quant_label_check.jfr");

        try (Recording rec = new Recording()) {
            rec.enable(MAT_VEC_EVENT);
            rec.start();

            GgufReader.QuantizedTensor A = twoByTwoMatrix();
            float[] x = {0f, 1f};
            LlamaTransformerHandler.matVec(A, x, 2, 2);

            rec.stop();
            rec.dump(jfrFile);
        }

        try (RecordingFile rf = new RecordingFile(jfrFile)) {
            while (rf.hasMoreEvents()) {
                RecordedEvent ev = rf.readEvent();
                if (!MAT_VEC_EVENT.equals(ev.getEventType().getName())) continue;
                String backend = ev.getString("backend");
                assertThat(backend)
                        .as("Backend label must not contain raw quant-type strings — "
                                + "those were the old broken labels that kept cpu.count=0")
                        .doesNotContain("quantized");
            }
        }
    }

    @Test
    @DisplayName("matVec(QuantizedTensor) result is numerically correct (F32 path)")
    void quantized_matVec_f32_numerical_correctness() {
        // A = [[1,2],[3,4]], x = [1,0] → y = [1,3]
        GgufReader.QuantizedTensor A = twoByTwoMatrix();
        float[] y = LlamaTransformerHandler.matVec(A, new float[]{1f, 0f}, 2, 2);
        assertThat(y[0]).isEqualTo(1f);
        assertThat(y[1]).isEqualTo(3f);

        // A = [[1,2],[3,4]], x = [0,1] → y = [2,4]
        float[] y2 = LlamaTransformerHandler.matVec(A, new float[]{0f, 1f}, 2, 2);
        assertThat(y2[0]).isEqualTo(2f);
        assertThat(y2[1]).isEqualTo(4f);
    }
}