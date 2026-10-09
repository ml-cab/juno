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
import static org.assertj.core.api.Assertions.within;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Context shift at the KV level: a session filled to {@code MAX_SEQ_LEN} drops
 * {@code discard} positions after the first {@code keep}, the rest move down
 * unchanged (V byte-exact, K through the caller's rotation), and writing goes on
 * where it would otherwise have thrown. Without a shift the cap still throws.
 */
@DisplayName("KvContextShift")
class KvContextShiftTest {

	private static final int KV_DIM = 4;
	private static final int MAX = DenseKvTensor.MAX_SEQ_LEN;
	private static final int KEEP = 5;
	private static final int DISCARD = (MAX - KEEP) / 2;
	private static final int PAGE = 16;

	static Stream<Arguments> tensors() {
		return Stream.of(
				Arguments.of("DenseKvTensor f16", (Supplier<SessionKvTensor>) () -> new DenseKvTensor(KvElementType.F16, KV_DIM)),
				Arguments.of("DenseKvTensor q8_0", (Supplier<SessionKvTensor>) () -> new DenseKvTensor(KvElementType.Q8_0, KV_DIM)),
				Arguments.of("PagedKvTensor f16", (Supplier<SessionKvTensor>) () -> new PagedKvTensor(
						new KvBlockPool(PAGE, KV_DIM, KvElementType.F16))),
				Arguments.of("PagedKvTensor q8_0", (Supplier<SessionKvTensor>) () -> new PagedKvTensor(
						new KvBlockPool(PAGE, KV_DIM, KvElementType.Q8_0))));
	}

	private static float[] row(int p) {
		return new float[] { p % 97, -(p % 89), (p % 13) * 0.5f, 1f };
	}

	private static SessionKvTensor filled(Supplier<SessionKvTensor> factory) {
		SessionKvTensor t = factory.get();
		for (int p = 0; p < MAX; p++)
			t.writeToken(p, row(p));
		return t;
	}

	private static float[] read(SessionKvTensor t, int pos) {
		float[] out = new float[KV_DIM];
		t.readToken(pos, out);
		return out;
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("tensors")
	@DisplayName("without a shift, a write at MAX_SEQ_LEN still throws")
	void capStillThrowsWithoutShift(String name, Supplier<SessionKvTensor> factory) {
		SessionKvTensor t = filled(factory);
		assertThatThrownBy(() -> t.writeToken(MAX, row(MAX))).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("MAX_SEQ_LEN");
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("tensors")
	@DisplayName("compact keeps the prefix, moves the tail down unchanged, and frees room to write")
	void compactMovesTailAndFreesRoom(String name, Supplier<SessionKvTensor> factory) {
		SessionKvTensor t = filled(factory);
		float[][] before = new float[MAX][];
		for (int p = 0; p < MAX; p++)
			before[p] = read(t, p);

		t.compact(KEEP, DISCARD, MAX);

		int newLen = MAX - DISCARD;
		for (int p = 0; p < KEEP; p++)
			assertThat(read(t, p)).as("kept prefix at %d", p).containsExactly(before[p]);
		for (int p = KEEP; p < newLen; p++)
			assertThat(read(t, p)).as("moved row at %d", p).containsExactly(before[p + DISCARD]);

		for (int p = newLen; p < MAX; p++)
			t.writeToken(p, row(p));
		assertThat(read(t, MAX - 1)[0]).isCloseTo(row(MAX - 1)[0], within(0.5f));
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("tensors")
	@DisplayName("shift rotates K of the moved rows only, and leaves V and the kept prefix alone")
	void shiftRotatesMovedKeysOnly(String name, Supplier<SessionKvTensor> factory) {
		SessionKvTensor k = filled(factory);
		SessionKvTensor v = filled(factory);
		float[][] kBefore = new float[MAX][];
		float[][] vBefore = new float[MAX][];
		for (int p = 0; p < MAX; p++) {
			kBefore[p] = read(k, p);
			vBefore[p] = read(v, p);
		}

		KvContextShift.shift(new SessionKvTensor[] { k }, new SessionKvTensor[] { v }, MAX, KEEP, DISCARD, r -> {
			for (int i = 0; i < r.length; i++)
				r[i] = -r[i];
		});

		int newLen = MAX - DISCARD;
		for (int p = 0; p < KEEP; p++) {
			assertThat(read(k, p)).containsExactly(kBefore[p]);
			assertThat(read(v, p)).containsExactly(vBefore[p]);
		}
		for (int p = KEEP; p < newLen; p++) {
			float[] expectK = kBefore[p + DISCARD];
			float[] gotK = read(k, p);
			for (int i = 0; i < KV_DIM; i++)
				assertThat(gotK[i]).as("K[%d][%d]", p, i).isCloseTo(-expectK[i], within(1e-4f));
			assertThat(read(v, p)).as("V[%d]", p).containsExactly(vBefore[p + DISCARD]);
		}
	}

	@Test
	@DisplayName("KvPageTable: compact truncates to the kept length and returns the pages past it")
	void pageTableReturnsPagesPastKeptLength() {
		KvBlockPool pool = new KvBlockPool(PAGE, KV_DIM, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		for (int p = 0; p < MAX; p++)
			table.appendToken(row(p));
		assertThat(pool.liveBlocks()).isEqualTo(MAX / PAGE);

		table.compact(KEEP, DISCARD);

		int newLen = MAX - DISCARD;
		assertThat(table.seqLen()).isEqualTo(newLen);
		assertThat(table.pageCount()).isEqualTo((newLen + PAGE - 1) / PAGE);
		assertThat(pool.liveBlocks()).isEqualTo(table.pageCount());
		table.appendToken(row(newLen));
		assertThat(table.seqLen()).isEqualTo(newLen + 1);
	}

	@ParameterizedTest(name = "{0}")
	@MethodSource("tensors")
	@DisplayName("out-of-range shift arguments are rejected")
	void rejectsBadArguments(String name, Supplier<SessionKvTensor> factory) {
		SessionKvTensor t = factory.get();
		for (int p = 0; p < 10; p++)
			t.writeToken(p, row(p));
		assertThatThrownBy(() -> t.compact(2, 0, 10)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> t.compact(-1, 2, 10)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> t.compact(5, 6, 10)).isInstanceOf(IllegalArgumentException.class);
	}
}
