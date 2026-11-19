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

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link SessionKvTensor#writeEncodedRows}: a window's rows handed over already in the tensor's stored blocks (where a
 * download of a quantized device KV lands) leave the tensor holding exactly the bytes that writing the same rows as
 * floats leaves, on every layout: the dense tensor, and the paged one across pages, past a gap, and over pages it
 * shares with another table.
 */
@DisplayName("SessionKvTensor.writeEncodedRows - a window's stored rows in one call")
class KvWriteEncodedRowsTest {

	private static final int KV_DIM = 96;
	private static final int PAGE = 16;
	private static final int ROWS = 37;
	private static final int START = 21;

	@ParameterizedTest
	@EnumSource(value = KvElementType.class, names = { "F16", "Q8_0", "Q4_0" })
	@DisplayName("dense: the stored bytes of a float write, positions before the window zero")
	void dense(KvElementType type) {
		check(new DenseKvTensor(type, KV_DIM), new DenseKvTensor(type, KV_DIM), START);
	}

	@ParameterizedTest
	@EnumSource(value = KvElementType.class, names = { "F16", "Q8_0", "Q4_0" })
	@DisplayName("paged: the stored bytes of a float write, across pages and past a gap")
	void paged(KvElementType type) {
		PagedKvArena arena = new PagedKvArena(PAGE, KV_DIM, type, type);
		check(arena.newK(), arena.newK(), START);
		check(arena.newK(), arena.newK(), 0);
		check(arena.newK(), arena.newK(), 2 * PAGE);
	}

	@ParameterizedTest
	@EnumSource(value = KvElementType.class, names = { "F16", "Q8_0", "Q4_0" })
	@DisplayName("dense: a reused tensor reads zero before the window, as after a float write")
	void denseReused(KvElementType type) {
		SessionKvLayout layout = SessionKvLayout.of(ServeScheduleOptions.defaults(), KvPageSizeOptions.defaults(),
				CacheTypeOptions.of(type, type), KV_DIM, null);
		SessionKvTensor[] first = layout.newKLayers(1);
		first[0].writeToken(0, rows(START, 11)[0]);
		first[0].writeToken(START - 1, rows(START, 11)[1]);
		layout.retireKLayers(first);
		SessionKvTensor t = layout.newKLayers(1)[0];
		float[][] window = rows(ROWS, 12);
		try (Arena arena = Arena.ofConfined()) {
			t.writeEncodedRows(START, ROWS, encoded(arena, type, window, 0), 0);
		}
		// Zero bytes, as a float write leaves them (a q4_0 zero block decodes to -0.0, so compare bytes, not floats).
		assertThat(stored(t, START + ROWS)).startsWith(new byte[START * KvRowCodec.rowBytes(type, KV_DIM)]);
	}

	@ParameterizedTest
	@EnumSource(value = KvElementType.class, names = { "F16", "Q8_0", "Q4_0" })
	@DisplayName("paged: a window over pages shared with another table writes copies, the other table unchanged")
	void pagedSharedPages(KvElementType type) {
		PagedKvArena arena = new PagedKvArena(PAGE, KV_DIM, type, type);
		float[][] prefix = rows(40, 21);
		PagedKvTensor source = arena.newK();
		PagedKvTensor reference = arena.newK();
		for (int p = 0; p < prefix.length; p++) {
			source.writeToken(p, prefix[p]);
			reference.writeToken(p, prefix[p]);
		}
		byte[] sourceBefore = stored(source, prefix.length);
		PagedKvTensor shared = source.sharePrefix(24);
		PagedKvTensor floatShared = reference.sharePrefix(24);
		float[][] window = rows(30, 22);
		int start = 20; // inside the second page, which both tables of each pair hold
		try (Arena a = Arena.ofConfined()) {
			shared.writeEncodedRows(start, window.length, encoded(a, type, window, 0), 0);
		}
		for (int r = 0; r < window.length; r++)
			floatShared.writeToken(start + r, window[r]);
		assertThat(shared.seqLen()).isEqualTo(start + window.length);
		assertThat(stored(shared, start + window.length)).containsExactly(stored(floatShared, start + window.length));
		assertThat(stored(source, prefix.length)).as("the table the pages were shared from").containsExactly(sourceBefore);
	}

	@Test
	@DisplayName("f32 tensors hold floats: refused, nothing written")
	void f32Refused() {
		DenseKvTensor dense = new DenseKvTensor(KvElementType.F32, KV_DIM);
		PagedKvTensor paged = new PagedKvArena(PAGE, KV_DIM, KvElementType.F32, KvElementType.F32).newK();
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment src = arena.allocate((long) ROWS * KV_DIM * Float.BYTES);
			assertThatThrownBy(() -> dense.writeEncodedRows(0, ROWS, src, 0))
					.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("f32");
			assertThatThrownBy(() -> paged.writeEncodedRows(0, ROWS, src, 0))
					.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("f32");
		}
		assertThat(dense.writtenTokens()).isZero();
		assertThat(paged.seqLen()).isZero();
	}

	/**
	 * Writes {@link #ROWS} rows at {@code start} into {@code bulk} as stored blocks (from an offset into native
	 * memory) and into {@code byFloat} a row at a time as floats, then compares every stored byte of
	 * {@code [0, start + ROWS)}.
	 */
	private static void check(SessionKvTensor bulk, SessionKvTensor byFloat, int start) {
		float[][] window = rows(ROWS, 7 + start);
		try (Arena arena = Arena.ofConfined()) {
			long offset = 5; // the window need not start the segment, nor sit on any alignment
			bulk.writeEncodedRows(start, ROWS, encoded(arena, bulk.type(), window, offset), offset);
		}
		for (int r = 0; r < ROWS; r++)
			byFloat.writeToken(start + r, window[r]);
		int len = start + ROWS;
		assertThat(stored(bulk, len)).containsExactly(stored(byFloat, len));
		assertThat(bulk.toFloatArray(len)).containsExactly(byFloat.toFloatArray(len));
	}

	/** {@code rows} encoded by the host codec, back to back, at {@code offset} into a new segment. */
	private static MemorySegment encoded(Arena arena, KvElementType type, float[][] rows, long offset) {
		int bpt = KvRowCodec.rowBytes(type, KV_DIM);
		byte[] packed = new byte[rows.length * bpt];
		for (int r = 0; r < rows.length; r++)
			KvRowCodec.encode(type, rows[r], 0, KV_DIM, packed, r * bpt);
		MemorySegment s = arena.allocate(offset + packed.length);
		MemorySegment.copy(packed, 0, s, JAVA_BYTE, offset, packed.length);
		return s;
	}

	/** The stored bytes of positions {@code [0, len)}, read through the attention view's pages. */
	private static byte[] stored(SessionKvTensor t, int len) {
		KvRows view = t.attentionRows(len);
		int bpr = view.bytesPerRow();
		byte[] out = new byte[len * bpr];
		for (int p = 0; p < len; p++) {
			int page = p / view.rowsPerPage();
			int slot = p % view.rowsPerPage();
			System.arraycopy(view.bytePage(page), slot * bpr, out, p * bpr, bpr);
		}
		return out;
	}

	private static float[][] rows(int n, long seed) {
		Random rnd = new Random(seed);
		float[][] out = new float[n][KV_DIM];
		for (float[] row : out)
			for (int i = 0; i < KV_DIM; i++)
				row[i] = rnd.nextFloat() * 4f - 2f;
		return out;
	}
}
