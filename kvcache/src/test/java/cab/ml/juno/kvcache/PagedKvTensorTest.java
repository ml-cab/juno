package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("PagedKvTensor")
class PagedKvTensorTest {

	@Test
	@DisplayName("F16 gather matches DenseKvTensor writes")
	void f16_matches_dense() {
		int kvDim = 8;
		int pageSize = 4;
		DenseKvTensor dense = new DenseKvTensor(KvElementType.F16, kvDim);
		PagedKvArena arena = new PagedKvArena(pageSize, kvDim, KvElementType.F16, KvElementType.F16);
		PagedKvTensor paged = arena.newK();

		float[] tok = new float[kvDim];
		for (int p = 0; p < 6; p++) {
			for (int i = 0; i < kvDim; i++)
				tok[i] = p * 10 + i;
			dense.writeToken(p, tok);
			paged.writeToken(p, tok);
		}
		float[] scratch = new float[6 * kvDim];
		float[] denseView = dense.viewForAttention(6, null);
		float[] pagedView = paged.viewForAttention(6, scratch);
		assertThat(pagedView).hasSize(6 * kvDim);
		for (int i = 0; i < 6 * kvDim; i++)
			assertThat(pagedView[i]).isEqualTo(denseView[i]);
		assertThat(paged.pageCount()).isEqualTo(2);
		assertThat(arena.kPool().allocatedBytes()).isEqualTo(2L * pageSize * kvDim * Float.BYTES);
	}

	@Test
	@DisplayName("Q8_0 pages use less memory than F16 and roundtrip")
	void q8_smaller_and_roundtrips() {
		int kvDim = 256;
		int tokens = 33;
		int pageSize = 16;
		PagedKvArena f16Arena = new PagedKvArena(pageSize, kvDim, KvElementType.F16, KvElementType.F16);
		PagedKvArena q8Arena = new PagedKvArena(pageSize, kvDim, KvElementType.Q8_0, KvElementType.Q8_0);
		PagedKvTensor f16 = f16Arena.newK();
		PagedKvTensor q8 = q8Arena.newK();

		float[] tok = new float[kvDim];
		for (int p = 0; p < tokens; p++) {
			for (int i = 0; i < kvDim; i++)
				tok[i] = (float) Math.sin(p * 0.1 + i * 0.01);
			f16.writeToken(p, tok);
			q8.writeToken(p, tok);
		}
		assertThat(q8Arena.kPool().allocatedBytes() * 2L)
				.isLessThanOrEqualTo(f16Arena.kPool().allocatedBytes());

		float[] scratch = new float[tokens * kvDim];
		float[] got = q8.viewForAttention(tokens, scratch);
		for (int i = 0; i < kvDim; i++)
			assertThat(got[i]).isCloseTo((float) Math.sin(i * 0.01), within(0.05f));
	}

	@Test
	@DisplayName("release returns pages; mid-write zero-fills holes like dense")
	void release_and_zero_fill_holes() {
		PagedKvArena arena = new PagedKvArena(4, 8, KvElementType.F16, KvElementType.F16);
		PagedKvTensor t = arena.newK();
		float[] tok = new float[8];
		for (int i = 0; i < 8; i++)
			tok[i] = i + 1;
		t.writeToken(0, tok);
		t.writeToken(2, tok); // fills pos 1 with zeros
		assertThat(t.seqLen()).isEqualTo(3);
		float[] view = t.viewForAttention(3, new float[24]);
		assertThat(view[8]).isZero(); // hole at pos 1
		assertThat(view[16]).isEqualTo(1f);
		t.release();
		assertThat(t.seqLen()).isZero();
		assertThat(arena.kPool().liveBlocks()).isZero();
	}
}
