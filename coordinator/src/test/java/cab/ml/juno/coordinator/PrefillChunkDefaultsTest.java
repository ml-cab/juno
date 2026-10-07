package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.function.IntToLongFunction;
import java.util.function.LongSupplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import cab.ml.juno.coordinator.PrefillChunkDefaults.Surface;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.ShardContext;

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

	// ── the handlers' prefill-window footprint ───────────────────────────────

	private static final long MIB = 1024L * 1024L;
	private static final IntToLongFunction ONE_MIB_PER_ROW = rows -> rows * MIB;

	@ParameterizedTest
	@EnumSource(value = Surface.class, names = { "LOCAL_REPL", "EMBEDDED" })
	void inProcess_static_gpu_sizes_from_the_window_footprint(Surface surface) {
		// 2 GiB free, 1 GiB headroom, 1 MiB a row.
		assertThat(PrefillChunkDefaults.resolveFrom(surface, null, true, GPU_WITH_HEADROOM, ONE_MIB_PER_ROW))
				.isEqualTo(1024);
	}

	@Test
	void continuous_schedule_ignores_the_footprint() {
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.LOCAL_REPL, null, false, GPU_WITH_HEADROOM,
				ONE_MIB_PER_ROW)).isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void windowBytesOf_sums_every_shard_that_holds_a_window() {
		IntToLongFunction f = PrefillChunkDefaults.windowBytesOf(List.of(handler(MIB), handler(0), handler(2 * MIB)));
		assertThat(f).isNotNull();
		assertThat(f.applyAsLong(100)).isEqualTo(300 * MIB);
	}

	@Test
	void windowBytesOf_is_absent_when_no_shard_runs_the_device_region() {
		assertThat(PrefillChunkDefaults.windowBytesOf(List.of(handler(0), handler(0)))).isNull();
		assertThat(PrefillChunkDefaults.windowBytesOf(List.of())).isNull();
	}

	// ── the KV mirror the handlers keep free for ─────────────────────────────

	@Test
	void the_window_is_sized_from_what_the_reserved_kv_mirror_leaves() {
		// 2 GiB free, of which 1 GiB is held for the KV mirror's growth: the window gets
		// half of the remaining 1 GiB, 512 rows at 1 MiB a row, and does not eat the mirror.
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.LOCAL_REPL, null, true, GPU_WITH_HEADROOM,
				ONE_MIB_PER_ROW, 1024 * MIB)).isEqualTo(512);
	}

	@Test
	void a_mirror_reserve_larger_than_free_memory_gives_the_smallest_window() {
		assertThat(PrefillChunkDefaults.resolveFrom(Surface.LOCAL_REPL, null, true, GPU_WITH_HEADROOM,
				ONE_MIB_PER_ROW, 4 * TWO_GIB)).isEqualTo(PrefillBatchOptions.DEFAULT_CHUNK_SIZE);
	}

	@Test
	void mirrorReserveOf_sums_every_shard() {
		assertThat(PrefillChunkDefaults.mirrorReserveOf(
				List.of(handler(MIB, 100 * MIB), handler(0, 0), handler(MIB, 200 * MIB)))).isEqualTo(300 * MIB);
		assertThat(PrefillChunkDefaults.mirrorReserveOf(List.of())).isZero();
	}

	/** A handler whose prefill windows cost {@code perRow} device bytes a row. */
	private static ForwardPassHandler handler(long perRow) {
		return handler(perRow, 0L);
	}

	/** As {@link #handler(long)}, holding {@code mirrorReserve} device bytes for its KV mirror. */
	private static ForwardPassHandler handler(long perRow, long mirrorReserve) {
		return new ForwardPassHandler() {
			@Override
			public ForwardResult forward(ForwardRequest request, ShardContext context) {
				throw new UnsupportedOperationException();
			}

			@Override
			public boolean isReady() {
				return true;
			}

			@Override
			public long prefillWindowDeviceBytes(int rows) {
				return rows * perRow;
			}

			@Override
			public long kvMirrorReserveDeviceBytes() {
				return mirrorReserve;
			}
		};
	}
}
