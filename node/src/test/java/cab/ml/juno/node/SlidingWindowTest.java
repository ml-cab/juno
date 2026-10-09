package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Which layers attend within a window, read from GGUF metadata: the width from
 * {@code <arch>.attention.sliding_window} and the layer pattern from
 * {@code <arch>.attention.sliding_window_pattern}, either a per-layer boolean
 * array or an integer period. Indexed by global layer, so a pipeline shard reads
 * the layers it holds.
 */
@DisplayName("SlidingWindow - window width and layer pattern from GGUF metadata")
class SlidingWindowTest {

	private static final int GEMMA_LAYERS = 42;

	@TempDir
	Path dir;

	private SlidingWindow read(Object... keyValues) throws IOException {
		Path gguf = MetadataOnlyGguf.write(dir, "m.gguf", MetadataOnlyGguf.keys(keyValues));
		try (GgufReader r = GgufReader.open(gguf)) {
			return SlidingWindow.read(r);
		}
	}

	/** The shape {@code gemma-4-E4B} writes: five windowed layers, then one global, repeated. */
	private static boolean[] gemmaPattern(int layers) {
		boolean[] p = new boolean[layers];
		for (int i = 0; i < layers; i++)
			p[i] = i % 6 != 5;
		return p;
	}

	private static List<Integer> globalLayers(SlidingWindow w, int layers) {
		List<Integer> global = new ArrayList<>();
		for (int i = 0; i < layers; i++)
			if (w.forLayer(i) == 0)
				global.add(i);
		return global;
	}

	@Test
	@DisplayName("no width key: no window on any layer, whatever the pattern says")
	void absentWidthIsNoWindow() throws IOException {
		SlidingWindow none = read("general.architecture", "llama", "llama.block_count", 4);
		assertThat(none.declared()).isFalse();
		assertThat(globalLayers(none, 4)).containsExactly(0, 1, 2, 3);

		SlidingWindow patternOnly = read("general.architecture", "gemma4", "gemma4.block_count", GEMMA_LAYERS,
				"gemma4.attention.sliding_window_pattern", gemmaPattern(GEMMA_LAYERS));
		assertThat(patternOnly.declared()).isFalse();
		assertThat(globalLayers(patternOnly, GEMMA_LAYERS)).hasSize(GEMMA_LAYERS);
	}

	@Test
	@DisplayName("a width of 0 is no window")
	void zeroWidthIsNoWindow() throws IOException {
		SlidingWindow w = read("general.architecture", "llama", "llama.block_count", 4, "llama.attention.sliding_window",
				0);
		assertThat(w.declared()).isFalse();
		assertThat(globalLayers(w, 4)).containsExactly(0, 1, 2, 3);
	}

	@Test
	@DisplayName("width present, pattern absent: every layer windowed (the uniform case)")
	void widthWithoutPatternIsUniform() throws IOException {
		SlidingWindow w = read("general.architecture", "llama", "llama.block_count", 32, "llama.attention.sliding_window",
				4096);
		assertThat(w.declared()).isTrue();
		assertThat(w.width()).isEqualTo(4096);
		for (int i = 0; i < 32; i++)
			assertThat(w.forLayer(i)).as("layer " + i).isEqualTo(4096);
	}

	@Test
	@DisplayName("boolean pattern: true layers windowed, false layers global (gemma-4-E4B's shape)")
	void booleanPatternNamesTheGlobalLayers() throws IOException {
		SlidingWindow w = read("general.architecture", "gemma4", "gemma4.block_count", GEMMA_LAYERS,
				"gemma4.attention.sliding_window", 512, "gemma4.attention.sliding_window_pattern",
				gemmaPattern(GEMMA_LAYERS));
		assertThat(globalLayers(w, GEMMA_LAYERS)).containsExactly(5, 11, 17, 23, 29, 35, 41);
		assertThat(w.forLayer(0)).isEqualTo(512);
		assertThat(w.forLayer(4)).isEqualTo(512);
		assertThat(w.forLayer(6)).isEqualTo(512);
	}

	@Test
	@DisplayName("integer period N: the last layer of each period is global, the same layers the array names")
	void integerPeriodMatchesTheArray() throws IOException {
		SlidingWindow period = read("general.architecture", "gemma4", "gemma4.block_count", GEMMA_LAYERS,
				"gemma4.attention.sliding_window", 512, "gemma4.attention.sliding_window_pattern", 6);
		assertThat(globalLayers(period, GEMMA_LAYERS)).containsExactly(5, 11, 17, 23, 29, 35, 41);

		SlidingWindow two = read("general.architecture", "llama", "llama.block_count", 4, "llama.attention.sliding_window",
				8, "llama.attention.sliding_window_pattern", 2);
		assertThat(globalLayers(two, 4)).containsExactly(1, 3);
	}

	@Test
	@DisplayName("period 1 is every layer global; period 0 is every layer windowed")
	void periodEdgeCases() throws IOException {
		SlidingWindow one = read("general.architecture", "llama", "llama.block_count", 3, "llama.attention.sliding_window",
				8, "llama.attention.sliding_window_pattern", 1);
		assertThat(globalLayers(one, 3)).containsExactly(0, 1, 2);
		SlidingWindow zero = read("general.architecture", "llama", "llama.block_count", 3, "llama.attention.sliding_window",
				8, "llama.attention.sliding_window_pattern", 0);
		assertThat(globalLayers(zero, 3)).isEmpty();
	}

	@Test
	@DisplayName("a pipeline shard reads the pattern at its global layer indices")
	void shardUsesGlobalLayerIndices() throws IOException {
		SlidingWindow w = read("general.architecture", "gemma4", "gemma4.block_count", GEMMA_LAYERS,
				"gemma4.attention.sliding_window", 512, "gemma4.attention.sliding_window_pattern",
				gemmaPattern(GEMMA_LAYERS));
		// Global layers 10..17: 11 and 17 are global, local 1 and 7.
		assertThat(w.forShard(10, 8)).containsExactly(512, 0, 512, 512, 512, 512, 512, 0);
		// The same layers from the start of the model are a different pattern: local indexing would be wrong.
		assertThat(w.forShard(0, 8)).containsExactly(512, 512, 512, 512, 512, 0, 512, 512);
		assertThat(SlidingWindow.NONE.forShard(10, 3)).containsExactly(0, 0, 0);
	}

	@Test
	@DisplayName("a malformed window fails at load, naming the key")
	void malformedWindowFailsClosed() {
		assertThatThrownBy(() -> read("general.architecture", "gemma4", "gemma4.block_count", GEMMA_LAYERS,
				"gemma4.attention.sliding_window", 512, "gemma4.attention.sliding_window_pattern", gemmaPattern(41)))
				.isInstanceOf(UnsupportedModelException.class).hasMessageContaining("sliding_window_pattern")
				.hasMessageContaining("41").hasMessageContaining("42");
		assertThatThrownBy(() -> read("general.architecture", "llama", "llama.block_count", 4,
				"llama.attention.sliding_window", 8, "llama.attention.sliding_window_pattern", new int[] { 1, 0, 1, 0 }))
				.isInstanceOf(UnsupportedModelException.class).hasMessageContaining("sliding_window_pattern");
		assertThatThrownBy(() -> read("general.architecture", "llama", "llama.block_count", 4,
				"llama.attention.sliding_window", -1)).isInstanceOf(UnsupportedModelException.class)
				.hasMessageContaining("llama.attention.sliding_window");
		assertThatThrownBy(() -> read("general.architecture", "llama", "llama.block_count", 4,
				"llama.attention.sliding_window", "512")).isInstanceOf(UnsupportedModelException.class)
				.hasMessageContaining("llama.attention.sliding_window");
		assertThatThrownBy(() -> read("general.architecture", "llama", "llama.block_count", 4,
				"llama.attention.sliding_window", 8, "llama.attention.sliding_window_pattern", -2))
				.isInstanceOf(UnsupportedModelException.class).hasMessageContaining("sliding_window_pattern");
		assertThatThrownBy(() -> read("general.architecture", "gemma4", "gemma4.attention.sliding_window", 512,
				"gemma4.attention.sliding_window_pattern", gemmaPattern(GEMMA_LAYERS)))
				.isInstanceOf(UnsupportedModelException.class).hasMessageContaining("block_count");
	}

	private static Path model(String file) {
		Path p = Path.of("models", file);
		return p.toFile().exists() ? p : Path.of("..", "models", file);
	}

	@Test
	@DisplayName("real files: Phi-3.5-mini is uniform at 262144; gemma-4-E4B is 512 with every sixth layer global")
	void realFiles() throws IOException {
		Path phi = model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		if (phi.toFile().exists()) {
			try (GgufReader r = GgufReader.open(phi)) {
				SlidingWindow w = SlidingWindow.read(r);
				assertThat(w.width()).isEqualTo(262144);
				assertThat(globalLayers(w, 32)).isEmpty();
			}
		}
		Path gemma = model("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf");
		assumeTrue(gemma.toFile().exists() || phi.toFile().exists(), "neither windowed model present");
		if (gemma.toFile().exists()) {
			try (GgufReader r = GgufReader.open(gemma)) {
				SlidingWindow w = SlidingWindow.read(r);
				assertThat(w.width()).isEqualTo(512);
				assertThat(globalLayers(w, GEMMA_LAYERS)).containsExactly(5, 11, 17, 23, 29, 35, 41);
			}
		}
		Path mistral = model("mistral-7b-instruct-v0.1-q4_k_m.gguf");
		if (mistral.toFile().exists()) {
			try (GgufReader r = GgufReader.open(mistral)) {
				assertThat(SlidingWindow.read(r).declared()).as("Mistral 7B v0.1 declares no window").isFalse();
			}
		}
	}
}
