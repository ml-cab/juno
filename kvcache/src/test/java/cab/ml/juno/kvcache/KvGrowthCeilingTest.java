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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A dense host KV layer told how many positions its request can reach grows toward that count: doubling as before,
 * never past it, so a 2,048-token prompt with 64 more tokens allocates 2,112 positions, not 4,096. A request that goes
 * past it doubles as before. The ceiling reaches a request's layers whether they exist when it is given or are made
 * after ({@link SessionKvLayout#reserve}); a reused tensor starts with none; paged layers, which grow a page at a time,
 * ignore it.
 */
@DisplayName("Dense KV growth ceiling - grow toward the request's length")
class KvGrowthCeilingTest {

	private static final int KV_DIM = 8;

	@Test
	@DisplayName("doubles up to the ceiling, then to it, never past it")
	void stopsAtTheCeiling() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F16, KV_DIM);
		t.limitGrowth(2112);
		t.ensureCapacity(2047);
		assertThat(t.capacityTokens()).isEqualTo(2048);
		t.ensureCapacity(2048);
		assertThat(t.capacityTokens()).isEqualTo(2112);
		t.ensureCapacity(2111);
		assertThat(t.capacityTokens()).isEqualTo(2112);
		assertThat(t.heapBytes()).isEqualTo(2112L * KV_DIM * 2);
	}

	@Test
	@DisplayName("without a ceiling it doubles, as before")
	void doublesWithoutOne() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F16, KV_DIM);
		t.ensureCapacity(2048);
		assertThat(t.capacityTokens()).isEqualTo(4096);
	}

	@Test
	@DisplayName("past the ceiling it doubles from where it is")
	void pastTheCeiling() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F32, KV_DIM);
		t.limitGrowth(100);
		t.ensureCapacity(64);
		assertThat(t.capacityTokens()).isEqualTo(100);
		t.ensureCapacity(100);
		assertThat(t.capacityTokens()).isEqualTo(200);
	}

	@Test
	@DisplayName("a ceiling under the capacity shrinks nothing; values survive every growth")
	void neverShrinks() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.Q8_0, 32);
		float[] row = new float[32];
		for (int i = 0; i < 32; i++)
			row[i] = i - 16;
		t.writeToken(0, row);
		t.ensureCapacity(200);
		t.limitGrowth(10);
		assertThat(t.capacityTokens()).isEqualTo(256);
		float[] got = new float[32];
		t.readToken(0, got);
		float[] want = new float[32];
		KvRowCodec.decode(KvElementType.Q8_0, encoded(row), 0, want, 0, 32);
		assertThat(got).isEqualTo(want);
	}

	@Test
	@DisplayName("the ceiling is clamped to the KV cap and must be positive")
	void bounds() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F16, KV_DIM);
		t.limitGrowth(DenseKvTensor.MAX_SEQ_LEN + 5000);
		t.ensureCapacity(DenseKvTensor.MAX_SEQ_LEN - 1);
		assertThat(t.capacityTokens()).isEqualTo(DenseKvTensor.MAX_SEQ_LEN);
		assertThatThrownBy(() -> t.limitGrowth(0)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("layers made after the reservation take it; the next request's do not")
	void reservedBeforeTheLayers() {
		SessionKvLayout layout = layout(ServeScheduleOptions.defaults(), null);
		layout.reserve("r1", 100, null, null);
		SessionKvTensor[] k = layout.newKLayers("r1", 2);
		SessionKvTensor[] v = layout.newVLayers("r1", 2);
		for (SessionKvTensor[] side : new SessionKvTensor[][] { k, v })
			for (SessionKvTensor t : side) {
				t.ensureCapacity(64);
				assertThat(t.capacityTokens()).isEqualTo(100);
			}
		SessionKvTensor next = layout.newKLayers("r1", 1)[0];
		next.ensureCapacity(64);
		assertThat(next.capacityTokens()).as("the reservation was taken by r1's layers").isEqualTo(128);
	}

	@Test
	@DisplayName("existing layers take a reservation at once")
	void reservedAfterTheLayers() {
		SessionKvLayout layout = layout(ServeScheduleOptions.defaults(), null);
		SessionKvTensor[] k = layout.newKLayers("r2", 1);
		SessionKvTensor[] v = layout.newVLayers("r2", 1);
		layout.reserve("r2", 300, k, v);
		k[0].ensureCapacity(256);
		v[0].ensureCapacity(256);
		assertThat(k[0].capacityTokens()).isEqualTo(300);
		assertThat(v[0].capacityTokens()).isEqualTo(300);
	}

	@Test
	@DisplayName("a reservation no layers took is dropped with forget")
	void forget() {
		SessionKvLayout layout = layout(ServeScheduleOptions.defaults(), null);
		layout.reserve("r3", 100, null, null);
		layout.forget("r3");
		SessionKvTensor t = layout.newKLayers("r3", 1)[0];
		t.ensureCapacity(64);
		assertThat(t.capacityTokens()).isEqualTo(128);
	}

	@Test
	@DisplayName("a reused dense tensor starts without its last request's ceiling")
	void reuseClears() {
		SessionKvLayout layout = layout(ServeScheduleOptions.defaults(), null);
		layout.reserve("a", 100, null, null);
		SessionKvTensor[] k = layout.newKLayers("a", 1);
		k[0].ensureCapacity(64);
		layout.retireKLayers(k);
		SessionKvTensor reused = layout.newKLayers("b", 1)[0];
		assertThat(reused).isSameAs(k[0]);
		reused.ensureCapacity(100);
		assertThat(reused.capacityTokens()).isEqualTo(200);
	}

	@Test
	@DisplayName("paged layers ignore the ceiling: they grow a page at a time")
	void pagedIgnores() {
		PagedKvArena arena = new PagedKvArena(16, KV_DIM, KvElementType.F16, KvElementType.F16);
		SessionKvLayout layout = layout(ServeScheduleOptions.of(ServeScheduleOptions.Mode.CONTINUOUS), arena);
		layout.reserve("p", 20, null, null);
		SessionKvTensor[] k = layout.newKLayers("p", 1);
		layout.newVLayers("p", 1);
		for (int p = 0; p < 40; p++)
			k[0].writeToken(p, new float[KV_DIM]);
		assertThat(k[0].capacityTokens()).isEqualTo(48);
	}

	private static SessionKvLayout layout(ServeScheduleOptions schedule, PagedKvArena arena) {
		return SessionKvLayout.of(schedule, KvPageSizeOptions.defaults(),
				CacheTypeOptions.of(KvElementType.F16, KvElementType.F16), KV_DIM, arena);
	}

	private static byte[] encoded(float[] row) {
		byte[] b = new byte[KvRowCodec.rowBytes(KvElementType.Q8_0, row.length)];
		KvRowCodec.encode(KvElementType.Q8_0, row, 0, row.length, b, 0);
		return b;
	}
}
