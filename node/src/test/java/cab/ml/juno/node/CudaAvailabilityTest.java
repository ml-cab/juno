package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * CudaAvailability tests.
 *
 * The no-GPU tests always run and verify that the detection logic does not
 * throw even when CUDA is absent. The GPU-tagged tests only run when a device
 * is present.
 */
@DisplayName("CudaAvailability")
class CudaAvailabilityTest {

	// ── Always-run: safe fallback on any machine ──────────────────────────────

	@Test
	@DisplayName("isAvailable() returns a boolean without throwing")
	void is_available_does_not_throw() {
		boolean result = CudaAvailability.isAvailable();
		// just assert it's a valid boolean — value depends on hardware
		assertThat(result == true || result == false).isTrue();
	}

	@Test
	@DisplayName("deviceCount() returns 0 when CUDA is unavailable")
	void device_count_is_zero_when_no_cuda() {
		assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA is present");
		assertThat(CudaAvailability.deviceCount()).isEqualTo(0);
	}

	@Test
	@DisplayName("deviceName() returns 'unavailable' when CUDA is absent")
	void device_name_is_unavailable_when_no_cuda() {
		assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA is present");
		assertThat(CudaAvailability.deviceName(0)).isEqualTo("unavailable");
	}

	@Test
	@DisplayName("vramBytes() returns 0 when CUDA is unavailable")
	void vram_bytes_is_zero_when_no_cuda() {
		assumeFalse(CudaAvailability.isAvailable(), "Skipping — CUDA is present");
		assertThat(CudaAvailability.vramBytes(0)).isEqualTo(0L);
	}

	// ── GPU-tagged: run on AWS / any CUDA machine ─────────────────────────────

	@Test
	@Tag("gpu")
	@DisplayName("isAvailable() returns true on a CUDA node")
	void is_available_true_on_cuda_node() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assertThat(CudaAvailability.isAvailable()).isTrue();
	}

	@Test
	@Tag("gpu")
	@DisplayName("deviceCount() >= 1 on a CUDA node")
	void device_count_at_least_one() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		assertThat(CudaAvailability.deviceCount()).isGreaterThanOrEqualTo(1);
	}

	@Test
	@Tag("gpu")
	@DisplayName("deviceName(0) is non-empty on a CUDA node")
	void device_name_non_empty() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		String name = CudaAvailability.deviceName(0);
		assertThat(name).isNotBlank().isNotEqualTo("unavailable");
		System.out.println("CUDA device 0: " + name);
	}

	@Test
	@Tag("gpu")
	@DisplayName("vramBytes(0) > 1 GB on a CUDA node")
	void vram_bytes_above_1gb() {
		assumeTrue(CudaAvailability.isAvailable(), "No CUDA — skipping");
		long vram = CudaAvailability.vramBytes(0);
		System.out.printf("VRAM device 0: %.1f GB%n", vram / 1e9);
		assertThat(vram).isGreaterThan(1_000_000_000L);
	}
}