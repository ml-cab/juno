package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.PrefixCache;

class PrefixCacheTest {

	private final PrefixCache cache = new PrefixCache();

	private final int[] systemPrompt = { 10, 20, 30, 40, 50 };
	private final int[] fullRequest = { 10, 20, 30, 40, 50, 60, 70 }; // same prefix + user tokens

	@Test
	void empty_cache_returns_no_hit() {
		PrefixCache.PrefixMatch match = cache.findLongestPrefix(systemPrompt);
		assertThat(match.isHit()).isFalse();
		assertThat(match.matchedTokens()).isEqualTo(0);
	}

	@Test
	void exact_prefix_match_returns_correct_length() {
		cache.cachePrefix(systemPrompt, systemPrompt.length, "cache-key-1");

		PrefixCache.PrefixMatch match = cache.findLongestPrefix(fullRequest);

		assertThat(match.isHit()).isTrue();
		assertThat(match.matchedTokens()).isEqualTo(5);
		assertThat(match.cacheKey()).isEqualTo("cache-key-1");
	}

	@Test
	void partial_prefix_returns_longest_cached_segment() {
		int[] partial = { 10, 20, 30 }; // only 3 tokens cached
		cache.cachePrefix(partial, 3, "partial-key");

		PrefixCache.PrefixMatch match = cache.findLongestPrefix(fullRequest);

		assertThat(match.isHit()).isTrue();
		assertThat(match.matchedTokens()).isEqualTo(3);
	}

	@Test
	void longer_cached_prefix_wins_over_shorter() {
		int[] short3 = { 10, 20, 30 };
		int[] long5 = { 10, 20, 30, 40, 50 };
		cache.cachePrefix(short3, 3, "short-key");
		cache.cachePrefix(long5, 5, "long-key");

		PrefixCache.PrefixMatch match = cache.findLongestPrefix(fullRequest);

		assertThat(match.matchedTokens()).isEqualTo(5);
		assertThat(match.cacheKey()).isEqualTo("long-key");
	}

	@Test
	void no_match_when_tokens_diverge_immediately() {
		cache.cachePrefix(systemPrompt, 5, "key-1");

		PrefixCache.PrefixMatch match = cache.findLongestPrefix(new int[] { 99, 88, 77 });

		assertThat(match.isHit()).isFalse();
	}

	@Test
	void invalidate_removes_cached_prefix() {
		cache.cachePrefix(systemPrompt, 5, "key-to-remove");
		cache.invalidate("key-to-remove");

		PrefixCache.PrefixMatch match = cache.findLongestPrefix(fullRequest);
		assertThat(match.isHit()).isFalse();
	}

	@Test
	void empty_token_array_returns_no_hit() {
		assertThat(cache.findLongestPrefix(new int[0]).isHit()).isFalse();
		assertThat(cache.findLongestPrefix(null).isHit()).isFalse();
	}
}
