package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class NgramDraftCacheTest {

	@Test
	void propose_on_empty_cache_returns_empty_draft() {
		NgramDraftCache cache = new NgramDraftCache(3);
		int[] draft = cache.propose(new int[] { 1, 2, 3, 4, 5 }, 4);
		assertThat(draft).isEmpty();
	}

	@Test
	void propose_with_context_shorter_than_n_returns_empty_draft() {
		NgramDraftCache cache = new NgramDraftCache(3);
		cache.observe(new int[] { 1, 2, 3, 4, 5, 1, 2, 3, 4 });
		int[] draft = cache.propose(new int[] { 1, 2 }, 4);
		assertThat(draft).isEmpty();
	}

	@Test
	void observe_then_propose_chains_repeated_ngram() {
		NgramDraftCache cache = new NgramDraftCache(3);
		// "1 2 3 -> 4", "2 3 4 -> 5" indexed from this sequence.
		cache.observe(new int[] { 1, 2, 3, 4, 5 });

		int[] draft = cache.propose(new int[] { 9, 9, 1, 2, 3 }, 2);

		assertThat(draft).containsExactly(4, 5);
	}

	@Test
	void propose_stops_at_first_miss() {
		NgramDraftCache cache = new NgramDraftCache(3);
		cache.observe(new int[] { 1, 2, 3, 4 }); // only "1 2 3 -> 4" indexed

		int[] draft = cache.propose(new int[] { 1, 2, 3 }, 5);

		// After the first drafted token (4), window becomes "2 3 4", which was
		// never observed -> lookup misses -> draft stops at length 1.
		assertThat(draft).containsExactly(4);
	}

	@Test
	void observe_is_incremental_not_reindexed_from_scratch() {
		NgramDraftCache cache = new NgramDraftCache(2);
		cache.observe(new int[] { 1, 2, 3 }); // indexes "1 2 -> 3"
		assertThat(cache.size()).isEqualTo(1);

		cache.observe(new int[] { 1, 2, 3, 4 }); // indexes only the new "2 3 -> 4"
		assertThat(cache.size()).isEqualTo(2);

		int[] draft = cache.propose(new int[] { 1, 2 }, 2);
		assertThat(draft).containsExactly(3, 4);
	}

	@Test
	void cache_evicts_oldest_entry_past_max_size() {
		NgramDraftCache cache = new NgramDraftCache(1);
		int maxEntries = 4096;

		// Observe maxEntries + 1 distinct 1-grams: token i -> token i+1, for
		// i in [0, maxEntries]. This indexes maxEntries+1 distinct keys one at a
		// time (n=1 means each new position after the first is a fresh key).
		int[] tokens = new int[maxEntries + 2];
		for (int i = 0; i < tokens.length; i++)
			tokens[i] = i;
		cache.observe(tokens);

		assertThat(cache.size()).isEqualTo(maxEntries);

		// The oldest key (token 0 -> token 1) must have been evicted; the most
		// recent key must still resolve.
		assertThat(cache.propose(new int[] { 0 }, 1)).isEmpty();
		int[] lastKey = { tokens.length - 2 };
		assertThat(cache.propose(lastKey, 1)).containsExactly(tokens.length - 1);
	}
}
