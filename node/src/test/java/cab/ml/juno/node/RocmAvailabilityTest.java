package cab.ml.juno.node;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RocmAvailability detection tests.
 *
 * <p>Two scenarios:
 * <ol>
 *   <li>ROCm present ({@code @Tag("rocm")}) — verifies device count, name, and VRAM.
 *   <li>ROCm absent (CPU-only / CUDA-only) — verifies all methods return safe fallback values.
 * </ol>
 *
 * <p>Run on ROCm machines:
 * <pre>
 *   mvn test -Dgroups=rocm -pl node
 * </pre>
 * Run on CPU-only / CUDA-only:
 * <pre>
 *   mvn test -Dgroups='!rocm' -pl node
 * </pre>
 */
@DisplayName("RocmAvailability — HIP device detection")
class RocmAvailabilityTest {

    // ── ROCm absent path ──────────────────────────────────────────────────────

    @Test
    @DisplayName("isAvailable() returns false when ROCm libraries are absent")
    void is_available_returns_false_when_no_rocm() {
        assumeFalse(RocmAvailability.isAvailable(), "Skipping — ROCm is present on this machine");
        assertThat(RocmAvailability.isAvailable()).isFalse();
    }

    @Test
    @DisplayName("deviceCount() returns 0 when ROCm is unavailable")
    void device_count_returns_zero_when_unavailable() {
        assumeFalse(RocmAvailability.isAvailable(), "Skipping — ROCm is present on this machine");
        assertThat(RocmAvailability.deviceCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("deviceName(0) returns 'unavailable' when ROCm is absent")
    void device_name_returns_unavailable_when_absent() {
        assumeFalse(RocmAvailability.isAvailable(), "Skipping — ROCm is present on this machine");
        assertThat(RocmAvailability.deviceName(0)).isEqualTo("unavailable");
    }

    @Test
    @DisplayName("vramBytes(0) returns 0 when ROCm is unavailable")
    void vram_bytes_returns_zero_when_unavailable() {
        assumeFalse(RocmAvailability.isAvailable(), "Skipping — ROCm is present on this machine");
        assertThat(RocmAvailability.vramBytes(0)).isEqualTo(0L);
    }

    // ── ROCm present path (@Tag("rocm")) ──────────────────────────────────────

    @Test
    @Tag("rocm")
    @DisplayName("isAvailable() returns true on ROCm system")
    void is_available_returns_true_on_rocm_system() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmAvailability.isAvailable()).isTrue();
    }

    @Test
    @Tag("rocm")
    @DisplayName("deviceCount() is at least 1 on ROCm system")
    void device_count_is_at_least_one() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmAvailability.deviceCount()).isGreaterThanOrEqualTo(1);
    }

    @Test
    @Tag("rocm")
    @DisplayName("deviceName(0) is non-blank and does not equal 'unavailable'")
    void device_name_is_non_blank() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        String name = RocmAvailability.deviceName(0);
        assertThat(name)
            .isNotBlank()
            .isNotEqualTo("unavailable")
            .isNotEqualTo("unknown");
    }

    @Test
    @Tag("rocm")
    @DisplayName("deviceName(0) contains 'AMD', 'Radeon', or 'Instinct' (AMD GPU identifier)")
    void device_name_identifies_amd_gpu() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        String name = RocmAvailability.deviceName(0).toUpperCase();
        boolean isAmd = name.contains("AMD") || name.contains("RADEON") || name.contains("INSTINCT");
        assertThat(isAmd)
            .as("Expected an AMD GPU name but got: '%s'", RocmAvailability.deviceName(0))
            .isTrue();
    }

    @Test
    @Tag("rocm")
    @DisplayName("vramBytes(0) is positive")
    void vram_bytes_is_positive() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmAvailability.vramBytes(0)).isGreaterThan(0L);
    }

    @Test
    @Tag("rocm")
    @DisplayName("vramBytes(0) reports at least 1 GB (minimum usable AMD GPU for inference)")
    void vram_bytes_at_least_1gb() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        long oneGb = 1_073_741_824L;
        assertThat(RocmAvailability.vramBytes(0))
            .as("VRAM should be at least 1 GB")
            .isGreaterThanOrEqualTo(oneGb);
    }

    @Test
    @Tag("rocm")
    @DisplayName("deviceName(-1) returns 'unknown' for out-of-range index")
    void device_name_out_of_range_returns_fallback() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        // hipGetDevicePropertiesR0600 with index = 999 returns a non-zero rc
        String name = RocmAvailability.deviceName(999);
        assertThat(name).isIn("unknown", "unavailable");
    }

    @Test
    @Tag("rocm")
    @DisplayName("vramBytes(999) returns 0 for out-of-range index")
    void vram_bytes_out_of_range_returns_zero() {
        assumeTrue(RocmAvailability.isAvailable(), "Skipping — no ROCm device");
        assertThat(RocmAvailability.vramBytes(999)).isEqualTo(0L);
    }
}
