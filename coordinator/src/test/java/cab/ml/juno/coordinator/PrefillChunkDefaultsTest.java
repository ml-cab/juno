package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.function.LongSupplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import cab.ml.juno.coordinator.PrefillChunkDefaults.Surface;

class PrefillChunkDefaultsTest {

	private static final long TWO_GIB = 2L * 1024 * 1024 * 1024;
	private static final LongSupplier GPU_WITH_HEADROOM = () -> TWO_GIB;
	private static final LongSupplier NO_GPU = null;
	private static final int ADAPTIVE_AT_TWO_GIB = PrefillBatchOptions.adaptiveChunkSize(TWO_GIB);

	@Test
	void localRepl_static_gpu_sizes_from_free_vram() {
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.LOCAL_REPL, null, true, GPU_WITH_HEADROOM))
				.isEqualTo(ADAPTIVE_AT_TWO_GIB);
	}

	@Test
	void embedded_static_gpu_sizes_from_free_vram() {
		// The JVM embedding facade runs the same in-process GPU pipeline as the local REPL.
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.EMBEDDED, null, true, GPU_WITH_HEADROOM))
				.isEqualTo(ADAPTIVE_AT_TWO_GIB);
	}

	@ParameterizedTest
	@EnumSource(value = Surface.class, names = { "LOCAL_REPL", "EMBEDDED" })
	void inProcess_surfaces_keep_the_fixed_default_on_cpu(Surface surface) {
		assertThat(PrefillChunkDefaults.resolveFrom(surface, null, true, NO_GPU))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@ParameterizedTest
	@EnumSource(value = Surface.class, names = { "LOCAL_REPL", "EMBEDDED" })
	void inProcess_surfaces_keep_the_fixed_default_under_continuous(Surface surface) {
		assertThat(PrefillChunkDefaults.resolveFrom(surface, null, false, GPU_WITH_HEADROOM))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@ParameterizedTest
	@EnumSource(value = Surface.class, names = { "CLUSTER_REPL", "COORDINATOR", "LORA_REPL" })
	void surfaces_without_a_meaningful_vram_query_keep_the_fixed_default(Surface surface) {
		assertThat(PrefillChunkDefaults.resolveFrom(surface, null, true, GPU_WITH_HEADROOM))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void failed_vram_query_falls_back_to_the_fixed_default() {
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.EMBEDDED, null, true, () -> 0L))
				.isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@ParameterizedTest
	@EnumSource(Surface.class)
	void explicit_value_overrides_on_every_surface_backend_and_schedule(Surface surface) {
		for (boolean staticSchedule : new boolean[] { true, false }) {
			assertThat(PrefillChunkDefaults.resolveFrom(surface, 7, staticSchedule, GPU_WITH_HEADROOM)).isEqualTo(7);
			assertThat(PrefillChunkDefaults.resolveFrom(surface, 7, staticSchedule, NO_GPU)).isEqualTo(7);
		}
	}
}
