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

import java.util.Arrays;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A released request's dense host KV tensors are handed to the next request instead
 * of allocating new ones, and a reused tensor never shows the earlier request's
 * values: every position it reads was written by the current request or reads zero,
 * as a new tensor's would.
 */
@DisplayName("SessionKvLayout - dense KV tensors reused across requests")
class SessionKvReuseTest {

	private static final int KV_DIM = 32; // one Q8_0 block

	private String prevSched;
	private String prevPage;
	private String prevK;
	private String prevV;

	@BeforeEach
	void save() {
		prevSched = System.getProperty(ServeScheduleOptions.ENV);
		prevPage = System.getProperty(KvPageSizeOptions.ENV);
		prevK = System.getProperty(CacheTypeOptions.ENV_K);
		prevV = System.getProperty(CacheTypeOptions.ENV_V);
		System.clearProperty(ServeScheduleOptions.ENV);
		System.clearProperty(KvPageSizeOptions.ENV);
		System.clearProperty(CacheTypeOptions.ENV_K);
		System.clearProperty(CacheTypeOptions.ENV_V);
	}

	@AfterEach
	void restoreAll() {
		restore(ServeScheduleOptions.ENV, prevSched);
		restore(KvPageSizeOptions.ENV, prevPage);
		restore(CacheTypeOptions.ENV_K, prevK);
		restore(CacheTypeOptions.ENV_V, prevV);
	}

	@Test
	@DisplayName("a retired session's tensors go to the next session, capacity kept")
	void retiredTensorsAreReused() {
		SessionKvLayout layout = SessionKvLayout.fromEnv(KV_DIM);
		SessionKvTensor[] k = layout.newKLayers(3);
		SessionKvTensor[] v = layout.newVLayers(3);
		for (SessionKvTensor t : k)
			t.ensureCapacity(511);
		layout.retireKLayers(k);
		layout.retireVLayers(v);

		SessionKvTensor[] k2 = layout.newKLayers(3);
		SessionKvTensor[] v2 = layout.newVLayers(3);
		for (int li = 0; li < 3; li++) {
			assertThat(k2[li]).isSameAs(k[li]);
			assertThat(v2[li]).isSameAs(v[li]);
		}
		assertThat(k2[0].capacityTokens()).isEqualTo(512);
		// Taken once: the next session allocates.
		assertThat(layout.newKLayers(3)[0]).isNotSameAs(k[0]);
	}

	@Test
	@DisplayName("F16: a reused tensor reads zero wherever the new session has not written")
	void reusedF16NeverShowsEarlierValues() {
		checkNoEarlierValues(SessionKvLayout.fromEnv(KV_DIM));
	}

	@Test
	@DisplayName("Q8_0: a reused tensor reads zero wherever the new session has not written")
	void reusedQ8NeverShowsEarlierValues() {
		System.setProperty(CacheTypeOptions.ENV_K, "q8_0");
		System.setProperty(CacheTypeOptions.ENV_V, "q8_0");
		checkNoEarlierValues(SessionKvLayout.fromEnv(KV_DIM));
	}

	private static void checkNoEarlierValues(SessionKvLayout layout) {
		SessionKvTensor[] first = layout.newKLayers(1);
		float[] ones = new float[KV_DIM];
		Arrays.fill(ones, 1f);
		for (int p = 0; p < 20; p++)
			first[0].writeToken(p, ones);
		layout.retireKLayers(first);

		SessionKvTensor t = layout.newKLayers(1)[0];
		assertThat(t).isSameAs(first[0]);
		float[] twos = new float[KV_DIM];
		Arrays.fill(twos, 2f);
		// The new session's first write lands past position 0 (a window at a later start).
		t.writeToken(5, twos);
		float[] row = new float[KV_DIM];
		for (int p = 0; p < 5; p++) {
			t.readToken(p, row);
			assertThat(row).as("position %d below the first write", p).containsOnly(0f);
		}
		t.readToken(5, row);
		float two = row[0]; // 2 as the element type stores it (Q8_0 rounds it)
		assertThat(two).isCloseTo(2f, org.assertj.core.data.Offset.offset(1e-3f));
		assertThat(row).containsOnly(two);

		// A view past what the session wrote reads zero there too.
		float[] scratch = new float[12 * KV_DIM];
		float[] view = t.viewForAttention(12, scratch);
		for (int p = 6; p < 12; p++)
			for (int d = 0; d < KV_DIM; d++)
				assertThat(view[p * KV_DIM + d]).as("position %d past the last write", p).isZero();
		assertThat(t.toFloatArray(20)).as("the earlier session's positions 12 to 19").containsOnly(0f, two);
	}

	@Test
	@DisplayName("the larger retired set is kept; a different layer count gets new tensors")
	void keepsTheLargerSet() {
		SessionKvLayout layout = SessionKvLayout.fromEnv(KV_DIM);
		SessionKvTensor[] big = layout.newKLayers(2);
		for (SessionKvTensor t : big)
			t.ensureCapacity(1023);
		SessionKvTensor[] small = layout.newKLayers(2);
		layout.retireKLayers(big);
		layout.retireKLayers(small);
		assertThat(layout.newKLayers(2)[0]).isSameAs(big[0]);

		layout.retireKLayers(big);
		assertThat(layout.newKLayers(3)[0]).isNotSameAs(big[0]);
	}

	@Test
	@DisplayName("paged layers are released to the arena, not reused")
	void pagedLayersAreReleased() {
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		System.setProperty(KvPageSizeOptions.ENV, "4");
		SessionKvLayout layout = SessionKvLayout.fromEnv(KV_DIM);
		PagedKvArena arena = layout.arena().orElseThrow();
		SessionKvTensor[] k = layout.newKLayers(1);
		k[0].writeToken(0, new float[KV_DIM]);
		int inUse = arena.kPool().liveBlocks();
		assertThat(inUse).isPositive();
		layout.retireKLayers(k);
		assertThat(arena.kPool().liveBlocks()).isLessThan(inUse);
		assertThat(layout.newKLayers(1)[0]).isNotSameAs(k[0]);
	}

	private static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
