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

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * {@link SessionKvTensor#attentionRows} hands attention the stored rows themselves: the pool's pages for a paged
 * tensor, the backing array for a dense one, never a copy. Every row read through the view equals
 * {@link SessionKvTensor#readToken} bit for bit, a row written after the view was taken is seen through it, and a
 * table that took a copy of a shared page (copy-on-write) reads its own page while the other reads the original.
 */
@DisplayName("KvRows - attention reads the stored rows in place")
class KvRowsTest {

	private static final int KV_DIM = 48; // not a multiple of the Q8_0 block, so the last block of a row is padded

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a paged view reads every written row as readToken does, across page sizes")
	void pagedRowsReadEveryRow(KvElementType type) {
		for (int pageSize : new int[] { 1, 3, 4, 16 }) {
			KvBlockPool pool = new KvBlockPool(pageSize, KV_DIM, type);
			PagedKvTensor neighbour = new PagedKvTensor(pool);
			PagedKvTensor t = new PagedKvTensor(pool);
			Random rng = new Random(7L * pageSize);
			for (int p = 0; p < 37; p++) {
				t.writeToken(p, row(rng));
				neighbour.writeToken(p, row(rng)); // the session's pages are not consecutive blocks
			}
			KvRows rows = t.attentionRows(37);
			assertThat(rows.type()).isEqualTo(type);
			assertThat(rows.rowsPerPage()).isEqualTo(pageSize);
			assertRowsEqualStored(rows, t, 37, "page " + pageSize);
		}
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a dense view is one page holding every row, and follows the backing array when it grows")
	void denseRowsAreOnePage(KvElementType type) {
		DenseKvTensor t = new DenseKvTensor(type, KV_DIM, 4);
		Random rng = new Random(11);
		for (int p = 0; p < 3; p++)
			t.writeToken(p, row(rng));
		KvRows rows = t.attentionRows(3);
		assertThat(rows.rowsPerPage()).isEqualTo(KvRows.ONE_PAGE);
		assertRowsEqualStored(rows, t, 3, "before growth");
		for (int p = 3; p < 70; p++) // grows the backing array past its first 4 rows twice
			t.writeToken(p, row(rng));
		assertRowsEqualStored(t.attentionRows(70), t, 70, "after growth");
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a row written after the view was taken is read through it: the view is not a copy")
	void viewIsNotACopy(KvElementType type) {
		Random rng = new Random(13);
		KvBlockPool pool = new KvBlockPool(4, KV_DIM, type);
		PagedKvTensor paged = new PagedKvTensor(pool);
		DenseKvTensor dense = new DenseKvTensor(type, KV_DIM);
		for (int p = 0; p < 6; p++) {
			float[] r = row(rng);
			paged.writeToken(p, r);
			dense.writeToken(p, r);
		}
		KvRows pagedRows = paged.attentionRows(6);
		KvRows denseRows = dense.attentionRows(6);
		float[] fresh = row(rng);
		paged.writeToken(5, fresh); // in place: page 1, slot 1, held once
		dense.writeToken(5, fresh);
		assertRowsEqualStored(pagedRows, paged, 6, "paged after an overwrite");
		assertRowsEqualStored(denseRows, dense, 6, "dense after an overwrite");
		float[] want = new float[KV_DIM];
		paged.readToken(5, want);
		assertThat(readRow(pagedRows, 5)).containsExactly(want);
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("tables sharing a prefix read the same page until one writes into it, then each its own")
	void sharedPagesAndCopyOnWrite(KvElementType type) {
		Random rng = new Random(17);
		KvBlockPool pool = new KvBlockPool(4, KV_DIM, type);
		KvPageTable source = new KvPageTable(pool);
		for (int p = 0; p < 6; p++)
			source.appendToken(row(rng));
		PagedKvTensor src = new PagedKvTensor(source);
		PagedKvTensor sharer = new PagedKvTensor(source.sharePrefix(6));

		KvRows srcRows = src.attentionRows(6);
		KvRows sharerRows = sharer.attentionRows(6);
		for (int page = 0; page < 2; page++)
			assertThat(samePage(srcRows, sharerRows, page)).as("page %d is shared, not copied", page).isTrue();

		sharer.writeToken(6, row(rng)); // into the shared, partly filled page 1: the sharer copies it first
		sharerRows = sharer.attentionRows(7);
		assertThat(samePage(srcRows, sharerRows, 0)).as("page 0 is still shared").isTrue();
		assertThat(samePage(srcRows, sharerRows, 1)).as("the sharer reads its own copy of page 1").isFalse();
		assertRowsEqualStored(src.attentionRows(6), src, 6, "source after the sharer's write");
		assertRowsEqualStored(sharerRows, sharer, 7, "sharer after its write");

		src.writeToken(6, row(rng)); // the source is now page 1's only holder and writes in place
		assertRowsEqualStored(src.attentionRows(7), src, 7, "source after its own write");
		assertRowsEqualStored(sharer.attentionRows(7), sharer, 7, "sharer unchanged by the source's write");
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("after truncate, a context shift and release, the view reads what the tensor holds")
	void viewFollowsTruncateShiftAndRelease(KvElementType type) {
		Random rng = new Random(19);
		KvBlockPool pool = new KvBlockPool(3, KV_DIM, type);
		PagedKvTensor t = new PagedKvTensor(pool);
		for (int p = 0; p < 14; p++)
			t.writeToken(p, row(rng));
		t.compact(2, 5, 12); // drops the draft rows 12 and 13, then rows [2, 7)
		assertThat(t.seqLen()).isEqualTo(7);
		assertRowsEqualStored(t.attentionRows(7), t, 7, "after the shift");
		for (int p = 7; p < 11; p++)
			t.writeToken(p, row(rng));
		assertRowsEqualStored(t.attentionRows(11), t, 11, "after appending past the shift");
		t.release();
		assertThat(pool.liveBlocks()).isZero();
		for (int p = 0; p < 5; p++)
			t.writeToken(p, row(rng));
		assertRowsEqualStored(t.attentionRows(5), t, 5, "reused after release");
	}

	@ParameterizedTest
	@EnumSource(KvElementType.class)
	@DisplayName("a reused dense tensor reads zero past what the new request wrote, never an earlier request's rows")
	void reusedDenseReadsZeroPastTheWrittenRows(KvElementType type) {
		DenseKvTensor t = new DenseKvTensor(type, KV_DIM);
		Random rng = new Random(23);
		for (int p = 0; p < 20; p++)
			t.writeToken(p, row(rng));
		t.reuse();
		for (int p = 0; p < 6; p++)
			t.writeToken(p, row(rng));
		KvRows rows = t.attentionRows(12);
		for (int p = 6; p < 12; p++)
			assertThat(readRow(rows, p)).as("position %d past the new request's writes", p).containsOnly(0f);
	}

	@Test
	@DisplayName("the view's length is checked against the rows written")
	void lengthIsChecked() {
		PagedKvTensor t = new PagedKvTensor(new KvBlockPool(4, KV_DIM, KvElementType.F16));
		t.writeToken(0, new float[KV_DIM]);
		assertThatThrownBy(() -> t.attentionRows(2)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> t.attentionRows(0)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("float rows already in one array are a one-page view of that array")
	void ofFloatArray() {
		float[] flat = new float[5 * KV_DIM];
		for (int i = 0; i < flat.length; i++)
			flat[i] = i;
		KvRows rows = KvRows.of(flat, KV_DIM);
		assertThat(rows.type()).isEqualTo(KvElementType.F16);
		assertThat(rows.rowsPerPage()).isEqualTo(KvRows.ONE_PAGE);
		assertThat(rows.floatPage(0)).isSameAs(flat);
	}

	/** Every row {@code [0, len)} read through the view equals the tensor's own readToken, bit for bit. */
	private static void assertRowsEqualStored(KvRows rows, SessionKvTensor t, int len, String what) {
		float[] want = new float[KV_DIM];
		for (int p = 0; p < len; p++) {
			t.readToken(p, want);
			float[] got = readRow(rows, p);
			for (int i = 0; i < KV_DIM; i++)
				if (Float.floatToRawIntBits(got[i]) != Float.floatToRawIntBits(want[i]))
					throw new AssertionError(what + ": row " + p + " element " + i + " reads " + got[i]
							+ " through the view, " + want[i] + " stored");
		}
	}

	/** Row {@code t} as attention reads it: page {@code t / rowsPerPage}, decoded in place for Q8_0. */
	private static float[] readRow(KvRows rows, int t) {
		int page = t / rows.rowsPerPage();
		int slot = t - page * rows.rowsPerPage();
		float[] out = new float[KV_DIM];
		if (rows.type() == KvElementType.F16)
			System.arraycopy(rows.floatPage(page), slot * KV_DIM, out, 0, KV_DIM);
		else
			Q8_0KvCodec.decodeRange(rows.bytePage(page), slot * rows.bytesPerRow(), 0, KV_DIM, out, 0);
		return out;
	}

	private static boolean samePage(KvRows a, KvRows b, int page) {
		return a.type() == KvElementType.F16 ? a.floatPage(page) == b.floatPage(page)
				: a.bytePage(page) == b.bytePage(page);
	}

	private static float[] row(Random rng) {
		float[] r = new float[KV_DIM];
		for (int i = 0; i < KV_DIM; i++)
			r[i] = (float) rng.nextGaussian();
		return r;
	}
}
