package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraCorpusLimit")
class LoraCorpusLimitTest {

	@Test
	@DisplayName("unlimited path keeps a single document unit")
	void unlimited_keeps_document_unit() {
		int[] tokens = seq(65); // 64 prediction positions
		boolean[] mask = allTrue(tokens.length);
		List<LoraTrainingLoop.TrainUnit> units = LoraCorpusLimit.limitDocument(tokens, mask, 32, 0, 42L);
		assertThat(units).hasSize(1);
		assertThat(units.get(0).tokens()).isEqualTo(tokens);
		assertThat(units.get(0).lossMask()).isEqualTo(mask);
	}

	@Test
	@DisplayName("within-budget path is a no-op")
	void within_budget_noop() {
		int[] tokens = seq(33); // 32 predictions
		boolean[] mask = allTrue(tokens.length);
		List<LoraTrainingLoop.TrainUnit> units = LoraCorpusLimit.limitDocument(tokens, mask, 32, 100, 7L);
		assertThat(units).hasSize(1);
		assertThat(units.get(0).tokens()).isEqualTo(tokens);
	}

	@Test
	@DisplayName("subsample respects supervised prediction budget")
	void subsample_respects_budget() {
		int[] tokens = seq(257); // 256 predictions → 8 chunks of 32
		boolean[] mask = allTrue(tokens.length);
		List<LoraTrainingLoop.TrainUnit> units = LoraCorpusLimit.limitDocument(tokens, mask, 32, 96, 42L);
		assertThat(units).isNotEmpty();
		int preds = predictionCount(units);
		assertThat(preds).isGreaterThanOrEqualTo(96);
		// Whole chunks only: each selected window has at most chunkTokens predictions
		assertThat(preds).isLessThanOrEqualTo(96 + 32 - 1);
		assertThat(units.size()).isLessThanOrEqualTo(4);
	}

	@Test
	@DisplayName("same seed selects the same chunks")
	void same_seed_deterministic() {
		int[] tokens = seq(257);
		boolean[] mask = allTrue(tokens.length);
		List<LoraTrainingLoop.TrainUnit> a = LoraCorpusLimit.limitDocument(tokens, mask, 32, 96, 42L);
		List<LoraTrainingLoop.TrainUnit> b = LoraCorpusLimit.limitDocument(tokens, mask, 32, 96, 42L);
		assertThat(unitFingerprints(a)).isEqualTo(unitFingerprints(b));
	}

	@Test
	@DisplayName("different seeds can select different chunks")
	void different_seeds_diverge() {
		int[] tokens = seq(257);
		boolean[] mask = allTrue(tokens.length);
		List<LoraTrainingLoop.TrainUnit> a = LoraCorpusLimit.limitDocument(tokens, mask, 32, 64, 1L);
		List<LoraTrainingLoop.TrainUnit> b = LoraCorpusLimit.limitDocument(tokens, mask, 32, 64, 2L);
		assertThat(unitFingerprints(a)).isNotEqualTo(unitFingerprints(b));
	}

	@Test
	@DisplayName("empty and short sequences return empty")
	void empty_and_short() {
		assertThat(LoraCorpusLimit.limitDocument(new int[0], new boolean[0], 32, 0, 1L)).isEmpty();
		assertThat(LoraCorpusLimit.limitDocument(new int[] { 1 }, new boolean[0], 32, 0, 1L)).isEmpty();
	}

	@Test
	@DisplayName("rejects invalid chunk and budget bounds")
	void rejects_invalid_bounds() {
		int[] tokens = seq(10);
		boolean[] mask = allTrue(tokens.length);
		assertThatThrownBy(() -> LoraCorpusLimit.limitDocument(tokens, mask, 0, 0, 1L))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraCorpusLimit.limitDocument(tokens, mask, 8193, 0, 1L))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraCorpusLimit.limitDocument(tokens, mask, 32, -1, 1L))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("validateChunkTokens accepts defaults and rejects ceiling")
	void validate_chunk_tokens() {
		assertThat(LoraCorpusLimit.validateChunkTokens(32)).isEqualTo(32);
		assertThat(LoraCorpusLimit.validateChunkTokens(128)).isEqualTo(128);
		assertThatThrownBy(() -> LoraCorpusLimit.validateChunkTokens(0))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraCorpusLimit.validateChunkTokens(LoraCorpusLimit.MAX_CHUNK_TOKENS + 1))
				.isInstanceOf(IllegalArgumentException.class);
	}

	private static int[] seq(int n) {
		int[] t = new int[n];
		for (int i = 0; i < n; i++)
			t[i] = i + 1;
		return t;
	}

	private static boolean[] allTrue(int tokenCount) {
		boolean[] m = new boolean[Math.max(0, tokenCount - 1)];
		Arrays.fill(m, true);
		return m;
	}

	private static int predictionCount(List<LoraTrainingLoop.TrainUnit> units) {
		int n = 0;
		for (var u : units)
			for (boolean m : u.lossMask())
				if (m)
					n++;
		return n;
	}

	private static Set<String> unitFingerprints(List<LoraTrainingLoop.TrainUnit> units) {
		Set<String> out = new HashSet<>();
		for (var u : units)
			out.add(Arrays.toString(u.tokens()) + "|" + Arrays.toString(u.lossMask()));
		return out;
	}
}
