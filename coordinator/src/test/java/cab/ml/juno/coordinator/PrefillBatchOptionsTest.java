package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class PrefillBatchOptionsTest {

	@Test
	void defaults_to_32() {
		assertThat(PrefillBatchOptions.defaults().chunkSize()).isEqualTo(32);
	}

	@Test
	void resolve_cli_overrides_default() {
		assertThat(PrefillBatchOptions.resolve(64).chunkSize()).isEqualTo(64);
	}

	@Test
	void of_rejects_zero() {
		assertThatThrownBy(() -> PrefillBatchOptions.of(0)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining(">= 1");
	}

	@Test
	void chunk_size_1_is_valid() {
		assertThat(PrefillBatchOptions.of(1).chunkSize()).isEqualTo(1);
	}

	@Test
	void resolveAdaptive_cli_overrides_even_with_gpu_context() {
		assertThat(PrefillBatchOptions.resolveAdaptive(64, null).chunkSize()).isEqualTo(64);
	}

	@Test
	void resolveAdaptive_falls_back_to_default_when_gpu_context_is_null() {
		assertThat(PrefillBatchOptions.resolveAdaptive(null, null).chunkSize())
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void adaptiveChunkSize_floor_is_the_fixed_default() {
		// Tiny headroom must never resolve below today's known-working fixed default —
		// the adaptive path can only grow the chunk, never shrink it below that floor.
		assertThat(PrefillBatchOptions.adaptiveChunkSize(1))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
		assertThat(PrefillBatchOptions.adaptiveChunkSize(0))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void adaptiveChunkSize_scales_with_free_vram() {
		// 2 GiB free * 0.5 headroom fraction / 65536 bytes-per-token ≈ 16384 tokens.
		long twoGiB = 2L * 1024 * 1024 * 1024;
		int expected = (int) ((twoGiB * PrefillBatchOptions.ADAPTIVE_HEADROOM_FRACTION)
				/ PrefillBatchOptions.ADAPTIVE_BYTES_PER_TOKEN);
		assertThat(PrefillBatchOptions.adaptiveChunkSize(twoGiB)).isEqualTo(expected);
	}

	@Test
	void adaptiveChunkSize_is_capped_at_ceiling() {
		long effectivelyUnlimited = Long.MAX_VALUE / 4;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(effectivelyUnlimited))
				.isEqualTo(PrefillBatchOptions.ADAPTIVE_CHUNK_CEILING);
	}
}
