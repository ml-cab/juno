package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.ref.WeakReference;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("KvPageReclaimer - idle KV pages released in the background")
class KvPageReclaimerTest {

	@Test
	@DisplayName("the period is read from its property: default, off, a value, and invalid values refused by name")
	void period_from_property() {
		assertThat(KvPageReclaimer.periodMs(null)).isEqualTo(KvPageReclaimer.DEFAULT_PERIOD_MS);
		assertThat(KvPageReclaimer.periodMs("  ")).isEqualTo(KvPageReclaimer.DEFAULT_PERIOD_MS);
		assertThat(KvPageReclaimer.periodMs("0")).isZero();
		assertThat(KvPageReclaimer.periodMs("1500")).isEqualTo(1500);
		for (String bad : new String[] { "-1", "abc", "1.5" })
			assertThatThrownBy(() -> KvPageReclaimer.periodMs(bad)).isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining(KvPageReclaimer.PERIOD_PROPERTY).hasMessageContaining(bad);
	}

	@Test
	@DisplayName("a scheduled pool has its idle pages released without any call from its users")
	void scheduled_pool_is_reclaimed_in_the_background() throws Exception {
		KvBlockPool pool = new KvBlockPool(4, 16, KvElementType.F16);
		int live = pool.allocate();
		for (int i = 0; i < 10; i++)
			pool.free(pool.allocate());
		assertThat(pool.storedPages()).isEqualTo(2);
		for (int i = 0; i < 9; i++)
			pool.allocate();
		int idle = pool.idlePages();
		assertThat(idle).isZero();
		// Ten pages held now; release nine and let the reclaimer find them idle.
		pool.free(live);
		AutoCloseable handle = KvPageReclaimer.schedule(pool, 20);
		try {
			long deadline = System.nanoTime() + 10_000_000_000L;
			while (pool.reclaimedPages() == 0 && System.nanoTime() < deadline)
				Thread.sleep(10);
		} finally {
			handle.close();
		}
		assertThat(pool.reclaimedPages()).isEqualTo(1);
		assertThat(pool.storedPages()).isEqualTo(9);
		assertThat(pool.liveBlocks()).isEqualTo(9);
	}

	@Test
	@DisplayName("a tick whose pool has been collected reports it, so its schedule ends")
	void tick_ends_once_the_pool_is_gone() {
		KvBlockPool pool = new KvBlockPool(4, 16, KvElementType.F16);
		WeakReference<KvBlockPool> ref = new WeakReference<>(pool);
		KvPageReclaimer.Tick tick = new KvPageReclaimer.Tick(ref);
		assertThat(tick.reclaimOnce()).isTrue();
		ref.clear();
		assertThat(tick.reclaimOnce()).isFalse();
	}

	@Test
	@DisplayName("register follows the property: 0 schedules nothing")
	void register_follows_the_property() {
		String prev = System.getProperty(KvPageReclaimer.PERIOD_PROPERTY);
		try {
			System.setProperty(KvPageReclaimer.PERIOD_PROPERTY, "0");
			assertThat(KvPageReclaimer.register(new KvBlockPool(4, 16, KvElementType.F16))).isFalse();
			System.setProperty(KvPageReclaimer.PERIOD_PROPERTY, "60000");
			assertThat(KvPageReclaimer.register(new KvBlockPool(4, 16, KvElementType.F16))).isTrue();
		} finally {
			if (prev == null)
				System.clearProperty(KvPageReclaimer.PERIOD_PROPERTY);
			else
				System.setProperty(KvPageReclaimer.PERIOD_PROPERTY, prev);
		}
	}
}
