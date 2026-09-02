package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.atomic.AtomicIntegerArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link SimdThreadPool}.
 *
 * <p>
 * {@link SimdThreadPool#POOL} is built once at class-init from whatever
 * {@code juno.simd.pool.size} happened to be set (or unset) when the JVM
 * started, so these tests do not attempt to exercise every value of that
 * property; that parsing logic is simple enough to review directly. What
 * they do verify is the property that actually matters for correctness:
 * {@link SimdThreadPool#forEachRow} must run the body exactly once for every
 * index in range and must not return until all of them have completed,
 * regardless of how the pool happens to be sized on the machine running the
 * test.
 */
@DisplayName("SimdThreadPool")
class SimdThreadPoolTest {

	@Test
	@DisplayName("forEachRow runs the body exactly once for every row in range")
	void forEachRow_visitsEveryRowExactlyOnce() {
		int rows = 500; // comfortably larger than any realistic pool parallelism
		AtomicIntegerArray visits = new AtomicIntegerArray(rows);

		SimdThreadPool.forEachRow(rows, r -> visits.incrementAndGet(r));

		for (int r = 0; r < rows; r++) {
			assertThat(visits.get(r)).as("row=%d", r).isEqualTo(1);
		}
	}

	@Test
	@DisplayName("forEachRow blocks until all rows complete before returning")
	void forEachRow_blocksUntilAllRowsComplete() {
		int rows = 200;
		AtomicIntegerArray visits = new AtomicIntegerArray(rows);

		SimdThreadPool.forEachRow(rows, r -> {
			// A tiny bit of artificial work so completion isn't trivially
			// instantaneous; still fast enough to keep the test quick.
			int sum = 0;
			for (int i = 0; i < 1000; i++) sum += i;
			visits.set(r, sum == 499500 ? 1 : -1);
		});

		// If forEachRow returned before every task finished, some slots
		// would still be at their initial value of 0.
		for (int r = 0; r < rows; r++) {
			assertThat(visits.get(r)).as("row=%d", r).isEqualTo(1);
		}
	}

	@Test
	@DisplayName("forEachRow with zero rows completes immediately without throwing")
	void forEachRow_zeroRows_doesNotThrow() {
		SimdThreadPool.forEachRow(0, r -> {
			throw new AssertionError("body must not be invoked for zero rows");
		});
	}

	@Test
	@DisplayName("diagnosticSummary reports the pool's actual parallelism and does not throw")
	void diagnosticSummary_reportsActualParallelism() {
		String summary = SimdThreadPool.diagnosticSummary();

		assertThat(summary).contains("parallelism=" + SimdThreadPool.POOL.getParallelism());
		assertThat(summary).contains("juno.simd.pool.size");
	}

	@Test
	@DisplayName("pool parallelism matches availableProcessors when the property was not set for this JVM")
	void poolParallelism_matchesFallback_whenPropertyUnset() {
		// This test only asserts the fallback behavior when the property was
		// genuinely absent for this JVM run; if a developer or CI job passed
		// -Djuno.simd.pool.size explicitly, the pool legitimately reflects
		// that override instead, so the assertion is skipped in that case
		// rather than asserting a false expectation.
		org.junit.jupiter.api.Assumptions.assumeTrue(System.getProperty("juno.simd.pool.size") == null,
				"juno.simd.pool.size was explicitly set for this JVM; fallback behavior is not applicable");

		assertThat(SimdThreadPool.POOL.getParallelism()).isEqualTo(Runtime.getRuntime().availableProcessors());
	}
}