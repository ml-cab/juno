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
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.KvBlockPool;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvPageTable;
import cab.ml.juno.kvcache.KvRows;
import cab.ml.juno.kvcache.PagedKvTensor;

/**
 * The oracle the block-table attention read is held to. Attention reads paged KV in place, page by page through
 * the block table ({@link PagedKvTensor#attentionRows}), and dense KV as one page; the oracle is the earlier path,
 * the pages gathered into one contiguous array ({@link KvPageTable#gather}, here through
 * {@link PagedKvTensor#toFloatArray}) and the attention math run over that. Over the same rows, every CPU attention
 * form ({@link GqaMath#attend}, the decode batch's {@link GqaMath#attendPooledAsSerial}, the prefill window's
 * {@link GqaMath#attendPooled} with and without the vector head) gives the oracle's output bits reading the pages in
 * place, and reading the dense tensor in place; and the gathered rows equal the dense tensor's rows.
 *
 * <p>The paged side is laid out the way a running server lays it out: K and V pools
 * shared with a neighbour session whose rows are written in between, so a session's
 * pages are not consecutive blocks; pages recycled from freed sessions, so a page's
 * slots past the written length hold stale rows; and page sizes that leave the last
 * page partly filled. Each context length is read as a prefix of the stored rows, as
 * the prefill window reads them row by row. K and V may be stored in different types
 * ({@code --cache-type-k} and {@code --cache-type-v} are separate), and the reader decodes
 * a {@code Q8_0} row inside the attention loop, one head's slice at a time.
 */
@DisplayName("Paged KV attention - gather oracle")
class PagedKvAttentionOracleTest {

	/** Attention shape: query heads, KV heads and head width. Every kvDim is a multiple of the Q8_0 block. */
	record Shape(int heads, int kvHeads, int headDim) {
		int kvDim() {
			return kvHeads * headDim;
		}

		int qDim() {
			return heads * headDim;
		}

		int gqaRatio() {
			return heads / kvHeads;
		}

		@Override
		public String toString() {
			return heads + "q/" + kvHeads + "kv/" + headDim;
		}
	}

	private static final Shape[] SHAPES = { new Shape(4, 4, 8), new Shape(8, 2, 16), new Shape(16, 2, 32) };
	private static final int[] PAGE_SIZES = { 1, 3, 4, 16 };
	private static final int N = 37;
	/** Window lengths: 0 is none; 5 starts mid-page for every page size above 1; 64 never binds at N. */
	private static final int[] WINDOWS = { 0, 1, 5, 64 };

	/** The forms CPU attention takes; the vector head only where the Vector API module is present. */
	enum Form {
		SERIAL, POOLED_AS_SERIAL, POOLED_SCALAR, POOLED_VECTOR
	}

	/** One session's K and V, stored twice: in shared pages and in dense tensors. */
	static final class TwinKv {
		final KvBlockPool kPool;
		final KvBlockPool vPool;
		final PagedKvTensor pagedK;
		final PagedKvTensor pagedV;
		final DenseKvTensor denseK;
		final DenseKvTensor denseV;
		final PagedKvTensor neighbourK;
		final PagedKvTensor neighbourV;
		final int kvDim;
		final Random rng;

		TwinKv(KvElementType type, int pageSize, int kvDim, long seed) {
			this(type, type, pageSize, kvDim, seed);
		}

		TwinKv(KvElementType kType, KvElementType vType, int pageSize, int kvDim, long seed) {
			this.kvDim = kvDim;
			this.rng = new Random(seed);
			kPool = new KvBlockPool(pageSize, kvDim, kType);
			vPool = new KvBlockPool(pageSize, kvDim, vType);
			churn(kPool);
			churn(vPool);
			pagedK = new PagedKvTensor(kPool);
			pagedV = new PagedKvTensor(vPool);
			neighbourK = new PagedKvTensor(kPool);
			neighbourV = new PagedKvTensor(vPool);
			denseK = new DenseKvTensor(kType, kvDim);
			denseV = new DenseKvTensor(vType, kvDim);
		}

		/** Leaves freed pages full of large stale rows, so the session's pages are recycled ones. */
		private void churn(KvBlockPool pool) {
			PagedKvTensor gone = new PagedKvTensor(pool);
			float[] stale = new float[kvDim];
			for (int p = 0; p < 3 * pool.pageSize() + 1; p++) {
				for (int i = 0; i < kvDim; i++)
					stale[i] = 1000f + (float) rng.nextGaussian() * 100f;
				gone.writeToken(p, stale);
			}
			gone.release();
		}

		/** Writes one fresh K and V row at {@code pos} on both sides, and a neighbour row in between. */
		void write(int pos) {
			float[] k = row();
			float[] v = row();
			pagedK.writeToken(pos, k);
			denseK.writeToken(pos, k);
			neighbourK.writeToken(neighbourK.seqLen(), row());
			pagedV.writeToken(pos, v);
			denseV.writeToken(pos, v);
			neighbourV.writeToken(neighbourV.seqLen(), row());
		}

		void writeThrough(int n) {
			for (int p = 0; p < n; p++)
				write(p);
		}

		float[] row() {
			float[] r = new float[kvDim];
			for (int i = 0; i < kvDim; i++)
				r[i] = (float) rng.nextGaussian();
			return r;
		}
	}

	static Stream<Arguments> typesAndPages() {
		List<Arguments> out = new ArrayList<>();
		for (KvElementType type : KvElementType.values())
			for (int pageSize : PAGE_SIZES)
				out.add(Arguments.of(type, pageSize));
		return out.stream();
	}

	/** Every (K type, V type) pair, mixed ones included, at every page size. */
	static Stream<Arguments> typePairsAndPages() {
		List<Arguments> out = new ArrayList<>();
		for (KvElementType kType : KvElementType.values())
			for (KvElementType vType : KvElementType.values())
				for (int pageSize : PAGE_SIZES)
					out.add(Arguments.of(kType, vType, pageSize));
		return out.stream();
	}

	@ParameterizedTest(name = "{0}, page {1}")
	@MethodSource("typesAndPages")
	@DisplayName("every prefix of the gathered pages equals the dense rows, bit for bit")
	void gatheredPrefixesEqualDenseRows(KvElementType type, int pageSize) {
		for (Shape s : SHAPES) {
			TwinKv kv = new TwinKv(type, pageSize, s.kvDim(), 11L * pageSize + s.kvDim());
			kv.writeThrough(N);
			for (int len = 1; len <= N; len++) {
				assertBitsEqual(kv.pagedK.toFloatArray(len), kv.denseK.toFloatArray(len), len * s.kvDim(),
						"K, " + s + ", len " + len);
				assertBitsEqual(kv.pagedV.toFloatArray(len), kv.denseV.toFloatArray(len), len * s.kvDim(),
						"V, " + s + ", len " + len);
			}
		}
	}

	@ParameterizedTest(name = "K {0}, V {1}, page {2}")
	@MethodSource("typePairsAndPages")
	@DisplayName("attention reading the pages in place equals attention over the gathered rows, every form, window and length")
	void attentionReadingPagesInPlaceEqualsGathered(KvElementType kType, KvElementType vType, int pageSize) {
		for (Shape s : SHAPES) {
			TwinKv kv = new TwinKv(kType, vType, pageSize, s.kvDim(), 13L * pageSize + s.kvDim());
			kv.writeThrough(N);
			assertAttentionEqual(kv, s, N, "");
		}
	}

	@ParameterizedTest(name = "{0}, page {1}")
	@MethodSource("typesAndPages")
	@DisplayName("after a context shift and a dropped draft, the pages read in place still attend as the gathered rows")
	void afterContextShiftAndDroppedDraft(KvElementType type, int pageSize) {
		Shape s = SHAPES[1];
		TwinKv kv = new TwinKv(type, pageSize, s.kvDim(), 17L * pageSize);
		kv.writeThrough(N);
		// Rows N..N+3 are a rejected draft window: written, then dropped by the compaction's seqLen.
		for (int p = N; p < N + 4; p++)
			kv.write(p);
		int keep = 4;
		int discard = 9;
		kv.pagedK.compact(keep, discard, N);
		kv.pagedV.compact(keep, discard, N);
		kv.denseK.compact(keep, discard, N);
		kv.denseV.compact(keep, discard, N);
		int len = N - discard;
		assertThat(kv.pagedK.seqLen()).isEqualTo(len);
		assertAttentionEqual(kv, s, len, "after shift: ");
		// Generation continues on the shifted cache, past at least one more page boundary.
		for (int p = len; p < len + pageSize + 2; p++)
			kv.write(p);
		assertAttentionEqual(kv, s, len + pageSize + 2, "after shift and append: ");
	}

	@ParameterizedTest(name = "{0}, page {1}")
	@MethodSource("typesAndPages")
	@DisplayName("a first write past position 0 leaves zero rows before it on both sides")
	void holesBeforeTheFirstWriteAreZeroOnBoth(KvElementType type, int pageSize) {
		Shape s = SHAPES[0];
		TwinKv kv = new TwinKv(type, pageSize, s.kvDim(), 19L * pageSize);
		int first = 7;
		for (int p = first; p < first + 9; p++)
			kv.write(p);
		float[] view = kv.pagedK.toFloatArray(first);
		for (int i = 0; i < first * s.kvDim(); i++)
			assertThat(view[i]).as("hole value %d", i).isZero();
		assertAttentionEqual(kv, s, first + 9, "holes: ");
	}

	/**
	 * The inputs are fit to catch a block-table read that walks the wrong pages. Without a
	 * window attention is a sum over (key, value) pairs and does not see their order, so
	 * there the check is a page read twice in place of another; under a window, which keys
	 * are read depends on their positions, so there two pages swapped is seen as well.
	 */
	@Test
	@DisplayName("the oracle's inputs see a page read twice, and a swapped page under a window")
	void aWrongPageChangesTheOutput() {
		Shape s = SHAPES[1];
		int pageSize = 4;
		int window = 5;
		TwinKv kv = new TwinKv(KvElementType.F16, pageSize, s.kvDim(), 23L);
		kv.writeThrough(N);
		float[] k = kv.pagedK.toFloatArray(N);
		float[] v = kv.pagedV.toFloatArray(N);
		float[] q = queries(s, 1, 29L)[0];
		int pageFloats = pageSize * s.kvDim();

		float[] whole = attendOnce(s, q, k, v, 0);
		float[] twiceK = k.clone();
		float[] twiceV = v.clone();
		System.arraycopy(k, pageFloats, twiceK, 0, pageFloats);
		System.arraycopy(v, pageFloats, twiceV, 0, pageFloats);
		assertThat(maxAbsDiff(attendOnce(s, q, twiceK, twiceV, 0), whole)).as("page 1 read in place of page 0")
				.isGreaterThan(1e-4);

		// The window reads rows [N - window, N): the last full page and the partial one after it.
		float[] windowed = attendOnce(s, q, k, v, window);
		int last = (N / pageSize - 1) * pageFloats;
		assertThat(maxAbsDiff(attendOnce(s, q, swapPages(k, last - pageFloats, last, pageFloats),
				swapPages(v, last - pageFloats, last, pageFloats), window), windowed))
				.as("a page under the window swapped with the one before it").isGreaterThan(1e-4);
		assertThat(attendOnce(s, q, swapPages(k, 0, pageFloats, pageFloats), swapPages(v, 0, pageFloats, pageFloats),
				window)).as("pages before the window are not read").containsExactly(windowed);
	}

	private static float[] attendOnce(Shape s, float[] q, float[] k, float[] v, int window) {
		float[] out = new float[s.qDim()];
		GqaMath.attend(q, k, v, N, out, new float[N], s.heads(), s.headDim(), s.gqaRatio(), s.kvDim(), window);
		return out;
	}

	private static float[] swapPages(float[] rows, int a, int b, int pageFloats) {
		float[] out = rows.clone();
		System.arraycopy(rows, a, out, b, pageFloats);
		System.arraycopy(rows, b, out, a, pageFloats);
		return out;
	}

	/**
	 * Attention at every context length {@code [1, len]}, every form and window: the pages read in place and the dense
	 * tensor read in place each give the oracle's bits, the attention over the pages gathered into one array.
	 */
	private static void assertAttentionEqual(TwinKv kv, Shape s, int len, String label) {
		float[][] qs = queries(s, len, 31L * len + s.kvDim());
		float[] oracleOut = new float[s.qDim()];
		float[] pagedOut = new float[s.qDim()];
		float[] denseOut = new float[s.qDim()];
		float[] scores = new float[len];
		for (int seqLen = 1; seqLen <= len; seqLen++) {
			float[] gk = kv.pagedK.toFloatArray(seqLen);
			float[] gv = kv.pagedV.toFloatArray(seqLen);
			KvRows pk = kv.pagedK.attentionRows(seqLen);
			KvRows pv = kv.pagedV.attentionRows(seqLen);
			KvRows dk = kv.denseK.attentionRows(seqLen);
			KvRows dv = kv.denseV.attentionRows(seqLen);
			float[] q = qs[seqLen - 1];
			for (int window : WINDOWS)
				for (Form form : Form.values()) {
					if (form == Form.POOLED_VECTOR && !GqaMath.VECTOR_HEAD)
						continue;
					String what = label + s + ", seqLen " + seqLen + ", window " + window + ", " + form;
					attend(form, q, KvRows.of(gk, s.kvDim()), KvRows.of(gv, s.kvDim()), seqLen, oracleOut, scores, s,
							window);
					attend(form, q, pk, pv, seqLen, pagedOut, scores, s, window);
					assertBitsEqual(pagedOut, oracleOut, s.qDim(), what + ", pages read in place");
					attend(form, q, dk, dv, seqLen, denseOut, scores, s, window);
					assertBitsEqual(denseOut, oracleOut, s.qDim(), what + ", dense read in place");
					attendFlat(form, q, gk, gv, seqLen, denseOut, scores, s, window);
					assertBitsEqual(denseOut, oracleOut, s.qDim(), what + ", float[] form");
				}
		}
	}

	private static void attend(Form form, float[] q, KvRows k, KvRows v, int seqLen, float[] out, float[] scores,
			Shape s, int window) {
		switch (form) {
		case SERIAL -> GqaMath.attend(q, k, v, seqLen, out, scores, s.heads(), s.headDim(), s.gqaRatio(), s.kvDim(),
				window);
		case POOLED_AS_SERIAL -> GqaMath.attendPooledAsSerial(q, k, v, seqLen, out, s.heads(), s.headDim(),
				s.gqaRatio(), s.kvDim(), window);
		case POOLED_SCALAR -> GqaMath.attendPooled(q, k, v, seqLen, out, s.heads(), s.headDim(), s.gqaRatio(),
				s.kvDim(), window, false);
		case POOLED_VECTOR -> GqaMath.attendPooled(q, k, v, seqLen, out, s.heads(), s.headDim(), s.gqaRatio(),
				s.kvDim(), window, true);
		}
	}

	/** The {@code float[]} forms training and the GPU oracle call: a one-page view of the arrays. */
	private static void attendFlat(Form form, float[] q, float[] k, float[] v, int seqLen, float[] out,
			float[] scores, Shape s, int window) {
		switch (form) {
		case SERIAL -> GqaMath.attend(q, k, v, seqLen, out, scores, s.heads(), s.headDim(), s.gqaRatio(), s.kvDim(),
				window);
		case POOLED_AS_SERIAL -> GqaMath.attendPooledAsSerial(q, k, v, seqLen, out, s.heads(), s.headDim(),
				s.gqaRatio(), s.kvDim(), window);
		case POOLED_SCALAR -> GqaMath.attendPooled(q, k, v, seqLen, out, s.heads(), s.headDim(), s.gqaRatio(),
				s.kvDim(), window, false);
		case POOLED_VECTOR -> GqaMath.attendPooled(q, k, v, seqLen, out, s.heads(), s.headDim(), s.gqaRatio(),
				s.kvDim(), window, true);
		}
	}

	private static float[][] queries(Shape s, int count, long seed) {
		Random rng = new Random(seed);
		float[][] qs = new float[count][s.qDim()];
		for (float[] q : qs)
			for (int i = 0; i < q.length; i++)
				q[i] = (float) rng.nextGaussian();
		return qs;
	}

	private static void assertBitsEqual(float[] actual, float[] expected, int n, String what) {
		for (int i = 0; i < n; i++)
			if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i]))
				throw new AssertionError(what + ": element " + i + " is " + actual[i] + ", the oracle reads " + expected[i]);
	}

	private static double maxAbsDiff(float[] a, float[] b) {
		double m = 0;
		for (int i = 0; i < a.length; i++)
			m = Math.max(m, Math.abs(a[i] - b[i]));
		return m;
	}
}
