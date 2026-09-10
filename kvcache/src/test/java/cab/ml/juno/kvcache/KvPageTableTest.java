package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KvPageTable")
class KvPageTableTest {

	@Test
	@DisplayName("append grows pages; memory scales with used tokens at page granularity")
	void append_scales_with_pages() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		assertThat(table.seqLen()).isZero();
		assertThat(table.pageCount()).isZero();

		float[] tok = new float[8];
		for (int p = 0; p < 9; p++) {
			for (int i = 0; i < 8; i++)
				tok[i] = p * 10 + i;
			table.appendToken(tok);
		}
		assertThat(table.seqLen()).isEqualTo(9);
		assertThat(table.pageCount()).isEqualTo(3); // ceil(9/4)
		assertThat(pool.liveBlocks()).isEqualTo(3);
		assertThat(pool.allocatedBytes()).isEqualTo(3L * 4 * 8 * Float.BYTES);
	}

	@Test
	@DisplayName("gather copies contiguous prefix matching written tokens")
	void gather_matches_written() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		float[] tok = new float[8];
		for (int p = 0; p < 6; p++) {
			for (int i = 0; i < 8; i++)
				tok[i] = p + i * 0.1f;
			table.appendToken(tok);
		}
		float[] workspace = new float[6 * 8];
		table.gather(workspace);
		for (int p = 0; p < 6; p++) {
			for (int i = 0; i < 8; i++)
				assertThat(workspace[p * 8 + i]).isCloseTo(p + i * 0.1f, offset(1e-5f));
		}
	}

	@Test
	@DisplayName("release returns all pages to the pool")
	void release_frees_pool() {
		KvBlockPool pool = new KvBlockPool(4, 8, KvElementType.F16);
		KvPageTable table = new KvPageTable(pool);
		float[] tok = new float[8];
		for (int p = 0; p < 5; p++)
			table.appendToken(tok);
		assertThat(pool.liveBlocks()).isEqualTo(2);
		table.release();
		assertThat(table.seqLen()).isZero();
		assertThat(table.pageCount()).isZero();
		assertThat(pool.liveBlocks()).isZero();
	}
}
