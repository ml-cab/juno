package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KvPageCounts - KV page figures summed over arenas")
class KvPageCountsTest {

	@Test
	@DisplayName("live, idle and reclaimed pages and stored bytes are summed over both pools of every arena")
	void sums() {
		PagedKvArena a = new PagedKvArena(4, 32, KvElementType.F16, KvElementType.Q8_0);
		PagedKvArena b = new PagedKvArena(4, 32, KvElementType.F16, KvElementType.F16);
		// a.K: 3 live, 2 idle. a.V: 1 live. b.K: 1 live, 1 reclaimed. b.V: none.
		for (int i = 0; i < 3; i++)
			a.kPool().allocate();
		a.kPool().free(a.kPool().allocate());
		a.kPool().free(a.kPool().allocate());
		a.kPool().free(a.kPool().allocate());
		int extra = a.kPool().allocate();
		int extra2 = a.kPool().allocate();
		a.kPool().free(extra);
		a.kPool().free(extra2);
		a.vPool().allocate();
		b.kPool().allocate();
		b.kPool().free(b.kPool().allocate());
		b.kPool().reclaimIdle();
		b.kPool().reclaimIdle();

		KvPageCounts c = KvPageCounts.sum(List.of(a, b));
		assertThat(c.livePages()).isEqualTo(5);
		assertThat(c.idlePages()).isEqualTo(2);
		assertThat(c.reclaimedPages()).isEqualTo(1);
		long f16Page = 4L * 32 * Short.BYTES;
		long q8Page = 4L * Q8_0KvCodec.encodedBytes(32);
		assertThat(c.storedBytes()).isEqualTo(5 * f16Page + q8Page + f16Page);
		assertThat(KvPageCounts.sum(List.of())).isEqualTo(KvPageCounts.NONE);
	}
}
