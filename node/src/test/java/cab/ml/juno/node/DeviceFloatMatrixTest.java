/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 */
package cab.ml.juno.node;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link DeviceFloatMatrix} upload and lifecycle (requires CUDA).
 */
@Tag("gpu")
@DisplayName("DeviceFloatMatrix — GPU upload")
class DeviceFloatMatrixTest {

    private static GpuContext ctx;

    @BeforeAll
    static void initCuda() {
        assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
        ctx = GpuContext.init(0);
    }

    @AfterAll
    static void teardown() {
        if (ctx != null) ctx.close();
    }

    @Test
    @DisplayName("upload rejects host length != rows * cols")
    void upload_rejects_mismatched_length() {
        float[] host = new float[10];
        assertThatThrownBy(() -> DeviceFloatMatrix.upload(ctx, host, 3, 3))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("rows*cols");
    }

    @Test
    @DisplayName("devicePointer throws after close")
    void device_pointer_throws_after_close() {
        float[] host = new float[4];
        for (int i = 0; i < 4; i++) host[i] = i * 0.1f;
        DeviceFloatMatrix d = DeviceFloatMatrix.upload(ctx, host, 2, 2);
        d.close();
        assertThatThrownBy(d::devicePointer).isInstanceOf(IllegalStateException.class);
    }
}
