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

/** The tokens a key's prefix entry holds: what a session save records as the positions its KV rows cover. */
class PrefixCacheEntryTest {

	@Test
	@DisplayName("a key's entry: its latest cached prefix, a copy; none before caching or after invalidation")
	void entryFollowsTheKeysLatestPrefix() {
		KVCacheManager kv = new KVCacheManager(new GpuKVCache(1L << 20), new CpuKVCache(4));
		assertThat(kv.prefixEntry("s")).isEmpty();

		kv.cachePrefix(new int[] { 1, 2, 3, 4, 5 }, 4, "s");
		kv.cachePrefix(new int[] { 1, 2, 9 }, 3, "other");
		assertThat(kv.prefixEntry("s")).hasValueSatisfying(t -> assertThat(t).containsExactly(1, 2, 3, 4));

		kv.prefixEntry("s").get()[0] = 77;
		assertThat(kv.prefixEntry("s")).hasValueSatisfying(t -> assertThat(t).containsExactly(1, 2, 3, 4));

		kv.cachePrefix(new int[] { 1, 2, 3, 4, 5, 6 }, 6, "s");
		assertThat(kv.prefixEntry("s")).hasValueSatisfying(t -> assertThat(t).containsExactly(1, 2, 3, 4, 5, 6));

		kv.invalidatePrefix("s");
		assertThat(kv.prefixEntry("s")).isEmpty();
		assertThat(kv.prefixEntry("other")).hasValueSatisfying(t -> assertThat(t).containsExactly(1, 2, 9));
	}
}
