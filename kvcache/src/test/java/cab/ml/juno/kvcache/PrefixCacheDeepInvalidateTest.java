package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Invalidating a long cached prefix walks one trie node per token. It must not
 * recurse per node: a session of a few thousand tokens, invalidated from a
 * request's virtual thread (whose stack is small), overflowed the stack.
 */
@DisplayName("PrefixCache - invalidating a long prefix")
class PrefixCacheDeepInvalidateTest {

	@Test
	@DisplayName("a 20000-token prefix is invalidated on a virtual thread; another key's prefix survives")
	void deepInvalidateOnVirtualThread() throws Exception {
		PrefixCache cache = new PrefixCache();
		int[] tokens = new int[20_000];
		for (int i = 0; i < tokens.length; i++)
			tokens[i] = i % 977;
		int[] other = { 5, 6, 7 };
		cache.cachePrefix(tokens, tokens.length, "long");
		cache.cachePrefix(other, other.length, "short");

		AtomicReference<Throwable> failure = new AtomicReference<>();
		Thread t = Thread.ofVirtual().start(() -> {
			try {
				cache.invalidate("long");
			} catch (Throwable e) {
				failure.set(e);
			}
		});
		t.join();
		assertThat(failure.get()).isNull();
		assertThat(cache.findLongestPrefix(tokens).isHit()).isFalse();
		assertThat(cache.findLongestPrefix(other).cacheKey()).isEqualTo("short");
	}
}
