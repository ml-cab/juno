/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link PrefixCache#findLongestPrefix(int[], String)}: a lookup on behalf of one KV key matches only the prefixes
 * that key cached, so a request never skips prefill over positions written under another key. Several keys may cache
 * the same tokens, and each keeps its entry until it is invalidated.
 */
class PrefixCacheOwnerTest {

	private static final int[] SHORT = { 10, 20, 30 };
	private static final int[] LONG = { 10, 20, 30, 40, 50 };
	private static final int[] REQUEST = { 10, 20, 30, 40, 50, 60, 70 };

	@Test
	@DisplayName("a lookup for one key matches only that key's prefixes, the longest of them")
	void matchesOnlyTheOwnersPrefixes() {
		PrefixCache cache = new PrefixCache();
		cache.cachePrefix(SHORT, SHORT.length, "a");
		cache.cachePrefix(LONG, LONG.length, "b");

		assertThat(cache.findLongestPrefix(REQUEST, "a")).isEqualTo(new PrefixCache.PrefixMatch(3, "a"));
		assertThat(cache.findLongestPrefix(REQUEST, "b")).isEqualTo(new PrefixCache.PrefixMatch(5, "b"));
		assertThat(cache.findLongestPrefix(REQUEST, "c").isHit()).isFalse();
		assertThat(cache.findLongestPrefix(REQUEST, null).isHit()).as("a request without a key").isFalse();

		cache.cachePrefix(LONG, LONG.length, "a");
		assertThat(cache.findLongestPrefix(REQUEST, "a")).as("a's longer prefix, cached later")
				.isEqualTo(new PrefixCache.PrefixMatch(5, "a"));
	}

	@Test
	@DisplayName("two keys caching the same tokens each keep their entry; invalidating one leaves the other")
	void sameTokensUnderTwoKeys() {
		PrefixCache cache = new PrefixCache();
		cache.cachePrefix(LONG, LONG.length, "a");
		cache.cachePrefix(LONG, LONG.length, "b");

		assertThat(cache.findLongestPrefix(REQUEST, "a")).isEqualTo(new PrefixCache.PrefixMatch(5, "a"));
		assertThat(cache.findLongestPrefix(REQUEST, "b")).isEqualTo(new PrefixCache.PrefixMatch(5, "b"));

		cache.invalidate("a");
		assertThat(cache.findLongestPrefix(REQUEST, "a").isHit()).isFalse();
		assertThat(cache.findLongestPrefix(REQUEST, "b")).isEqualTo(new PrefixCache.PrefixMatch(5, "b"));
		assertThat(cache.findLongestPrefix(REQUEST)).as("the lookup across keys sees b's entry")
				.isEqualTo(new PrefixCache.PrefixMatch(5, "b"));

		cache.invalidate("b");
		assertThat(cache.findLongestPrefix(REQUEST).isHit()).isFalse();
	}

	@Test
	@DisplayName("invalidating a key keeps another key's shorter prefix on the same path")
	void invalidateKeepsAShorterPrefixOfAnotherKey() {
		PrefixCache cache = new PrefixCache();
		cache.cachePrefix(SHORT, SHORT.length, "a");
		cache.cachePrefix(LONG, LONG.length, "b");

		cache.invalidate("b");

		assertThat(cache.findLongestPrefix(REQUEST, "b").isHit()).isFalse();
		assertThat(cache.findLongestPrefix(REQUEST, "a")).isEqualTo(new PrefixCache.PrefixMatch(3, "a"));
	}

	@Test
	@DisplayName("caching again under a key replaces its entry: a branch drops the earlier path past the branch")
	void cachingAgainReplacesTheEntry() {
		PrefixCache cache = new PrefixCache();
		cache.cachePrefix(LONG, LONG.length, "a");
		cache.cachePrefix(LONG, LONG.length, "b");
		int[] branch = { 10, 20, 30, 41, 51, 61 };

		cache.cachePrefix(branch, branch.length, "a");

		assertThat(cache.findLongestPrefix(REQUEST, "a").isHit())
				.as("a's positions 3 and 4 now hold the branch's tokens, not 40 and 50").isFalse();
		assertThat(cache.findLongestPrefix(new int[] { 10, 20, 30, 41, 51, 61, 71 }, "a"))
				.isEqualTo(new PrefixCache.PrefixMatch(6, "a"));
		assertThat(cache.findLongestPrefix(REQUEST, "b")).isEqualTo(new PrefixCache.PrefixMatch(5, "b"));

		cache.invalidate("a");
		cache.invalidate("b");
		assertThat(cache.findLongestPrefix(REQUEST).isHit()).isFalse();
		assertThat(cache.findLongestPrefix(branch).isHit()).isFalse();
	}

	@Test
	@DisplayName("a lookup for a key counts as a lookup, and as a hit only when that key's prefix matches")
	void statsCountOnlyTheOwnersHits() {
		PrefixCache cache = new PrefixCache();
		cache.cachePrefix(LONG, LONG.length, "a");

		cache.findLongestPrefix(REQUEST, "b");
		cache.findLongestPrefix(REQUEST, null);
		assertThat(cache.lookupCount()).isEqualTo(2);
		assertThat(cache.hitCount()).isZero();

		cache.findLongestPrefix(REQUEST, "a");
		assertThat(cache.lookupCount()).isEqualTo(3);
		assertThat(cache.hitCount()).isEqualTo(1);
	}

	@Test
	@DisplayName("the longest common prefix matches the opening of a longer entry, and names a key whose entry runs"
			+ " through it")
	void commonPrefixMatchesTheOpeningOfALongerEntry() {
		PrefixCache cache = new PrefixCache();
		int[] system = { 1, 2, 3, 4 };
		int[] a = { 1, 2, 3, 4, 10, 11 };
		int[] b = { 1, 2, 3, 4, 20 };
		cache.cachePrefix(a, a.length, "a");

		assertThat(cache.findLongestPrefix(new int[] { 1, 2, 3, 4, 30 }).isHit()).as("no entry ends at 4").isFalse();
		PrefixCache.PrefixMatch m = cache.findLongestCommonPrefix(new int[] { 1, 2, 3, 4, 30 });
		assertThat(m.matchedTokens()).isEqualTo(4);
		assertThat(m.cacheKey()).isEqualTo("a");

		cache.cachePrefix(b, b.length, "b");
		cache.invalidate("a");
		m = cache.findLongestCommonPrefix(a);
		assertThat(m.matchedTokens()).as("a's branch is pruned; b runs through the opening").isEqualTo(system.length);
		assertThat(m.cacheKey()).isEqualTo("b");

		cache.invalidate("b");
		assertThat(cache.findLongestCommonPrefix(a).isHit()).isFalse();
	}
}

