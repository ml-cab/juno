package cab.ml.juno.health;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Duration;

import org.junit.jupiter.api.Test;

class CircuitBreakerTest {

	private CircuitBreaker cb() {
		return CircuitBreaker.forNode("n1");
	}

	private CircuitBreaker fast() {
		// Tiny window + instant wait — for transition tests
		return new CircuitBreaker("n1", 4, 0.50, Duration.ofMillis(1));
	}

	@Test
	void starts_closed() {
		assertThat(cb().state()).isEqualTo(CircuitState.CLOSED);
	}

	@Test
	void calls_permitted_when_closed() {
		assertThat(cb().isCallPermitted()).isTrue();
	}

	@Test
	void opens_after_failure_rate_exceeds_threshold() {
		CircuitBreaker breaker = fast();
		// 4 calls, 3 failures = 75% > 50%
		breaker.recordSuccess();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();

		assertThat(breaker.state()).isEqualTo(CircuitState.OPEN);
		assertThat(breaker.isCallPermitted()).isFalse();
	}

	@Test
	void stays_closed_below_failure_threshold() {
		CircuitBreaker breaker = fast();
		// 4 calls, 1 failure = 25% < 50%
		breaker.recordSuccess();
		breaker.recordSuccess();
		breaker.recordSuccess();
		breaker.recordFailure();

		assertThat(breaker.state()).isEqualTo(CircuitState.CLOSED);
	}

	@Test
	void transitions_to_half_open_after_wait_elapses() throws InterruptedException {
		CircuitBreaker breaker = fast();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure(); // open

		Thread.sleep(10); // wait > 1ms
		assertThat(breaker.isCallPermitted()).isTrue();
		assertThat(breaker.state()).isEqualTo(CircuitState.HALF_OPEN);
	}

	@Test
	void half_open_closes_on_success() throws InterruptedException {
		CircuitBreaker breaker = fast();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();

		Thread.sleep(10);
		breaker.isCallPermitted(); // transitions to HALF_OPEN
		breaker.recordSuccess();

		assertThat(breaker.state()).isEqualTo(CircuitState.CLOSED);
	}

	@Test
	void half_open_reopens_on_failure() throws InterruptedException {
		CircuitBreaker breaker = fast();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();

		Thread.sleep(10);
		breaker.isCallPermitted(); // HALF_OPEN
		breaker.recordFailure();

		assertThat(breaker.state()).isEqualTo(CircuitState.OPEN);
	}

	@Test
	void force_open_overrides_current_state() {
		CircuitBreaker breaker = cb();
		assertThat(breaker.state()).isEqualTo(CircuitState.CLOSED);
		breaker.forceOpen();
		assertThat(breaker.state()).isEqualTo(CircuitState.OPEN);
		assertThat(breaker.isCallPermitted()).isFalse();
	}

	@Test
	void reset_closes_and_clears_window() {
		CircuitBreaker breaker = fast();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();
		assertThat(breaker.state()).isEqualTo(CircuitState.OPEN);

		breaker.reset();

		assertThat(breaker.state()).isEqualTo(CircuitState.CLOSED);
		assertThat(breaker.isCallPermitted()).isTrue();
	}

	@Test
	void does_not_open_before_window_is_full() {
		CircuitBreaker breaker = fast(); // window=4
		// Only 3 failures — window not full yet
		breaker.recordFailure();
		breaker.recordFailure();
		breaker.recordFailure();

		assertThat(breaker.state()).isEqualTo(CircuitState.CLOSED);
	}
}
