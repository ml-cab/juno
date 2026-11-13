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

import java.lang.ref.WeakReference;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Releases idle KV pages in the background: every period, each registered {@link KvBlockPool} gives the heap back
 * the storage of the pages that stayed unused since the previous tick ({@link KvBlockPool#reclaimIdle}). The ticks
 * run on one daemon thread shared by every pool, never on a request's thread, so no decode step waits for them
 * beyond one short hold of the pool's lock per {@link KvBlockPool#RECLAIM_CHUNK} pages.
 *
 * <p>The period is {@value #PERIOD_PROPERTY} in milliseconds, {@value #DEFAULT_PERIOD_MS} by default; {@code 0}
 * turns the reclaim off, and any other value that is not a whole number of milliseconds is refused. A pool is held
 * weakly: once nothing else holds it, its schedule ends.
 */
public final class KvPageReclaimer {

	/** System property: the reclaim period in milliseconds; {@code 0} turns it off. */
	public static final String PERIOD_PROPERTY = "juno.kv.reclaimPeriodMs";
	/** One minute: pages unused for one to two minutes are released. */
	public static final long DEFAULT_PERIOD_MS = 60_000;

	private KvPageReclaimer() {
	}

	/** The period {@value #PERIOD_PROPERTY} sets now. */
	public static long periodMs() {
		return periodMs(System.getProperty(PERIOD_PROPERTY));
	}

	/** {@code value} read as a period: blank is the default, {@code 0} is off. */
	static long periodMs(String value) {
		if (value == null || value.isBlank())
			return DEFAULT_PERIOD_MS;
		long ms;
		try {
			ms = Long.parseLong(value.strip());
		} catch (NumberFormatException e) {
			ms = -1;
		}
		if (ms < 0)
			throw new IllegalArgumentException("-D" + PERIOD_PROPERTY + "=" + value
					+ " is not a period in milliseconds; use a whole number >= 0 (0 turns the KV page reclaim off)");
		return ms;
	}

	/**
	 * Schedules {@code pool} at the period {@value #PERIOD_PROPERTY} sets. Returns false, scheduling nothing, when
	 * the reclaim is off.
	 */
	public static boolean register(KvBlockPool pool) {
		long period = periodMs();
		if (period == 0)
			return false;
		schedule(pool, period);
		return true;
	}

	/** Schedules {@code pool} every {@code periodMs}; closing the handle ends the schedule. */
	static AutoCloseable schedule(KvBlockPool pool, long periodMs) {
		if (pool == null)
			throw new IllegalArgumentException("pool must not be null");
		if (periodMs < 1)
			throw new IllegalArgumentException("periodMs must be >= 1");
		Tick tick = new Tick(new WeakReference<>(pool));
		AtomicReference<ScheduledFuture<?>> self = new AtomicReference<>();
		ScheduledFuture<?> f = Ticker.EXEC.scheduleWithFixedDelay(() -> {
			if (!tick.reclaimOnce()) {
				ScheduledFuture<?> mine = self.get();
				if (mine != null)
					mine.cancel(false);
			}
		}, periodMs, periodMs, TimeUnit.MILLISECONDS);
		self.set(f);
		return () -> f.cancel(false);
	}

	/** One pool's tick. */
	static final class Tick {
		private final WeakReference<KvBlockPool> pool;

		Tick(WeakReference<KvBlockPool> pool) {
			this.pool = pool;
		}

		/** Reclaims the pool's idle pages; false once the pool is gone, so the schedule ends. */
		boolean reclaimOnce() {
			KvBlockPool p = pool.get();
			if (p == null)
				return false;
			p.reclaimIdle();
			return true;
		}
	}

	/** The shared daemon thread, started on first use. */
	private static final class Ticker {
		static final ScheduledExecutorService EXEC = create();

		private static ScheduledExecutorService create() {
			ScheduledThreadPoolExecutor e = new ScheduledThreadPoolExecutor(1, r -> {
				Thread t = new Thread(r, "juno-kv-page-reclaim");
				t.setDaemon(true);
				return t;
			});
			e.setRemoveOnCancelPolicy(true);
			return e;
		}
	}
}
