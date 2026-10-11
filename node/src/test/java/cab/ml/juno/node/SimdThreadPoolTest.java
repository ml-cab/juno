package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.atomic.AtomicIntegerArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link SimdThreadPool}: {@link SimdThreadPool#forEachRow} runs
 * the body exactly once for every index in range and does not return until all
 * of them have completed, on {@link CpuThreads}.
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
	@DisplayName("diagnosticSummary reports the CPU matmul thread count")
	void diagnosticSummary_reportsCpuThreads() {
		String summary = SimdThreadPool.diagnosticSummary();
		assertThat(summary).contains("CPU matmul threads: " + CpuThreads.threads());
		assertThat(summary).contains("--threads");
	}
}
