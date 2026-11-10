package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * {@link SlidingWindowTrainedLiveTest} on CUDA: the Phi-3 handler with its GPU
 * attention, on the model trained with a 2047-token window.
 * <ul>
 * <li>The KV oracle at width 2047 on a 3,000-token prompt, under each schedule:
 * the prompt is prefilled on the device, then each request's device KV is
 * retired and the next token attends on the host rows the oracle overwrote.</li>
 * <li>Teacher-forced NLL over 3,000 tokens with the file's window against none:
 * each position is one decode-width forward, so every token past position 2047
 * attends on the device under the window. The two must agree before 2047 and the
 * windowed run must read the lower NLL past it.</li>
 * </ul>
 * Opt-in like the CPU test: {@code -Djuno.test.trainedWindow=true}, a heap of
 * about 32 GB; about nine minutes on the reference host. Skips without a CUDA
 * device or the file.
 */
@Tag("gpu")
@EnabledIfSystemProperty(named = "juno.test.trainedWindow", matches = "true")
@DisplayName("Sliding-window attention - a model trained with a binding window, CUDA")
class SlidingWindowTrainedGpuTest {

	private static GpuContext gpu;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		gpu = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (gpu != null)
			gpu.close();
	}

	/** Loads on a fresh CUDA backend, recording it for release and whether GPU attention came up. */
	private static final class Loads implements SlidingWindowLiveCheck.Loader {
		private final Path path;
		private final ShardContext shard;
		final List<CudaMatVec> backends = new ArrayList<>();
		final List<Boolean> gpuAttention = new ArrayList<>();

		Loads(Path path, ShardContext shard) {
			this.path = path;
			this.shard = shard;
		}

		@Override
		public ForwardPassHandler load() throws Exception {
			CudaMatVec backend = new CudaMatVec(gpu);
			backends.add(backend);
			ForwardPassHandler h = Phi3TransformerHandler.load(path, shard, backend);
			gpuAttention.add(h.gpuAttentionActive());
			return h;
		}

		void release() {
			for (CudaMatVec b : backends)
				b.releaseScratch();
		}
	}

	@ParameterizedTest(name = "schedule {0}")
	@ValueSource(strings = { "static", "continuous" })
	@DisplayName("KV oracle at width 2047 on a 3,000-token prompt, device prefill")
	void kvOracle(String schedule) throws Exception {
		Path path = SlidingWindowTrainedLiveTest.present();
		LlamaConfig cfg = SlidingWindowTrainedLiveTest.config(path);
		ShardContext shard = SlidingWindowTrainedLiveTest.shard(cfg);
		Loads loads = new Loads(path, shard);
		try {
			SlidingWindowLiveCheck.Result r = SlidingWindowTrainedLiveTest.underSchedule(schedule,
					() -> SlidingWindowLiveCheck.run(loads, shard, cfg.numLayers(), cfg.kvDim(),
							SlidingWindowTrainedLiveTest.N, SlidingWindowTrainedLiveTest.WIDTH));
			System.out.printf(Locale.ROOT,
					"TRAINED-WINDOW-ORACLE cuda %s: windowed %.6f global %.4f effect %.4f gpu attention %s%n", schedule,
					r.windowedDiff(), r.globalDiff(), r.windowEffect(), loads.gpuAttention);
			assertThat(loads.gpuAttention.get(0)).as("GPU attention active on the windowed handler").isTrue();
			SlidingWindowTrainedLiveTest.assertOracle(r);
		} finally {
			loads.release();
		}
	}

	@Test
	@DisplayName("Teacher-forced NLL over 3,000 tokens on the device: the file's window reads lower NLL past 2047")
	void windowLowersNllPastItsWidth() throws Exception {
		Path path = SlidingWindowTrainedLiveTest.present();
		ShardContext shard = SlidingWindowTrainedLiveTest.shard(SlidingWindowTrainedLiveTest.config(path));
		Loads loads = new Loads(path, shard);
		try {
			SlidingWindowTrainedLiveTest.assertWindowLowersNll("cuda", loads, shard);
			System.out.println("TRAINED-WINDOW-NLL cuda gpu attention " + loads.gpuAttention);
			assertThat(loads.gpuAttention).as("GPU attention active on both handlers").containsOnly(true);
		} finally {
			loads.release();
		}
	}
}
