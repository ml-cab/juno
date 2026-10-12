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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link SessionKvTensor#writeRows}: a window's rows written in one call straight from
 * native memory (where a device download lands) hold exactly what writing them one
 * token at a time from Java arrays holds, on every layout.
 */
@DisplayName("SessionKvTensor.writeRows - a window's rows from native memory in one call")
class KvWriteRowsTest {

	private static final int KV_DIM = 64;
	private static final int ROWS = 37;
	private static final int START = 21;

	@Test
	@DisplayName("dense F16 equals token-at-a-time writes")
	void denseF16() {
		check(new DenseKvTensor(KvElementType.F16, KV_DIM), new DenseKvTensor(KvElementType.F16, KV_DIM));
	}

	@Test
	@DisplayName("dense Q8_0 equals token-at-a-time writes")
	void denseQ8() {
		check(new DenseKvTensor(KvElementType.Q8_0, KV_DIM), new DenseKvTensor(KvElementType.Q8_0, KV_DIM));
	}

	@Test
	@DisplayName("paged equals token-at-a-time writes")
	void paged() {
		PagedKvArena arena = new PagedKvArena(16, KV_DIM, KvElementType.F16, KvElementType.F16);
		check(arena.newK(), arena.newK());
	}

	@Test
	@DisplayName("on a reused tensor, the positions below the window read zero")
	void reusedTensorGapReadsZero() {
		SessionKvLayout layout = SessionKvLayout.of(ServeScheduleOptions.defaults(), KvPageSizeOptions.defaults(),
				CacheTypeOptions.defaults(), KV_DIM, null);
		SessionKvTensor[] first = layout.newKLayers(1);
		float[] ones = new float[KV_DIM];
		java.util.Arrays.fill(ones, 1f);
		for (int p = 0; p < START + ROWS; p++)
			first[0].writeToken(p, ones);
		layout.retireKLayers(first);
		SessionKvTensor t = layout.newKLayers(1)[0];
		try (Arena arena = Arena.ofConfined()) {
			t.writeRows(START, ROWS, rows(arena, new Random(3)), 0);
		}
		float[] row = new float[KV_DIM];
		for (int p = 0; p < START; p++) {
			t.readToken(p, row);
			assertThat(row).as("position %d", p).containsOnly(0f);
		}
	}

	private static void check(SessionKvTensor bulk, SessionKvTensor byToken) {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment src = rows(arena, new Random(7));
			long offset = 8L * Float.BYTES; // the window need not start the segment
			MemorySegment shifted = arena.allocate(src.byteSize() + offset, Float.BYTES);
			MemorySegment.copy(src, 0, shifted, offset, src.byteSize());
			bulk.writeRows(START, ROWS, shifted, offset);
			float[] row = new float[KV_DIM];
			for (int r = 0; r < ROWS; r++) {
				MemorySegment.copy(src, JAVA_FLOAT, (long) r * KV_DIM * Float.BYTES, row, 0, KV_DIM);
				byToken.writeToken(START + r, row);
			}
		}
		assertThat(bulk.toFloatArray(START + ROWS)).containsExactly(byToken.toFloatArray(START + ROWS));
	}

	private static MemorySegment rows(Arena arena, Random rnd) {
		MemorySegment s = arena.allocate((long) ROWS * KV_DIM * Float.BYTES, Float.BYTES);
		for (long i = 0; i < (long) ROWS * KV_DIM; i++)
			s.setAtIndex(JAVA_FLOAT, i, rnd.nextFloat() * 4f - 2f);
		return s;
	}
}
