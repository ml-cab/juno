package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.function.IntToLongFunction;

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

	// ── sized from the prefill regions' real window footprint ────────────────

	private static final long MIB = 1024L * 1024L;
	private static final long TWO_GIB = 2048 * MIB;

	@Test
	void adaptiveChunkSize_takes_the_widest_window_whose_footprint_fits_the_headroom() {
		// 2 GiB free, half of it headroom: 1 GiB at 1 MiB a row is 1024 rows, not the
		// 16384 the fixed per-token figure would give.
		IntToLongFunction oneMiBPerRow = rows -> rows * MIB;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, oneMiBPerRow)).isEqualTo(1024);
	}

	@Test
	void adaptiveChunkSize_handles_a_footprint_that_grows_with_the_square_of_the_window() {
		// Attention scores grow with rows x context, and a window starting the prompt has
		// a context of its own width: 1024 x 1024 x 1 KiB is exactly the 1 GiB headroom.
		IntToLongFunction quadratic = rows -> (long) rows * rows * 1024L;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, quadratic)).isEqualTo(1024);
	}

	@Test
	void adaptiveChunkSize_divides_the_headroom_between_shards_that_each_hold_a_window() {
		// In-process shards keep one window each on the same card.
		IntToLongFunction twoShards = rows -> 2 * rows * MIB;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, twoShards)).isEqualTo(512);
	}

	@Test
	void adaptiveChunkSize_with_a_footprint_keeps_the_fixed_default_as_its_floor() {
		IntToLongFunction huge = rows -> rows * TWO_GIB;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, huge))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void adaptiveChunkSize_with_a_footprint_is_capped_at_ceiling() {
		IntToLongFunction tiny = rows -> rows;
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, tiny))
				.isEqualTo(PrefillBatchOptions.ADAPTIVE_CHUNK_CEILING);
	}

	@Test
	void adaptiveChunkSize_without_a_region_keeps_the_per_token_figure() {
		// No handler runs prefill windows on the device region (zero bytes, or no
		// function at all): the host-staged path's per-token figure still applies.
		int perToken = PrefillBatchOptions.adaptiveChunkSize(TWO_GIB);
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, rows -> 0L)).isEqualTo(perToken);
		assertThat(PrefillBatchOptions.adaptiveChunkSize(TWO_GIB, null)).isEqualTo(perToken);
	}

	@Test
	void resolveAdaptiveFrom_uses_the_footprint_when_given() {
		assertThat(PrefillBatchOptions.resolveAdaptiveFrom(null, () -> TWO_GIB, rows -> rows * MIB).chunkSize())
				.isEqualTo(1024);
		assertThat(PrefillBatchOptions.resolveAdaptiveFrom(64, () -> TWO_GIB, rows -> rows * MIB).chunkSize())
				.isEqualTo(64);
	}
}
