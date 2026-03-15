package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.RetryPolicy;

class RetryPolicyTest {

	// ── Validation ─────────────────────────────────────────────────────────

	@Test
	void rejects_zero_max_attempts() {
		assertThatThrownBy(() -> RetryPolicy.of(0, 0)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("maxAttempts");
	}

	@Test
	void rejects_negative_backoff() {
		assertThatThrownBy(() -> RetryPolicy.of(2, -1)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("backoffMs");
	}

	// ── Presets ────────────────────────────────────────────────────────────

	@Test
	void none_has_one_attempt_and_no_retry() {
		RetryPolicy p = RetryPolicy.none();
		assertThat(p.maxAttempts()).isEqualTo(1);
		assertThat(p.backoffMs()).isEqualTo(0);
		assertThat(p.hasRetry()).isFalse();
	}

	@Test
	void once_has_two_attempts() {
		RetryPolicy p = RetryPolicy.once();
		assertThat(p.maxAttempts()).isEqualTo(2);
		assertThat(p.hasRetry()).isTrue();
	}

	@Test
	void aggressive_has_three_attempts() {
		RetryPolicy p = RetryPolicy.aggressive();
		assertThat(p.maxAttempts()).isEqualTo(3);
		assertThat(p.hasRetry()).isTrue();
	}

	@Test
	void custom_policy_round_trips() {
		RetryPolicy p = RetryPolicy.of(5, 250);
		assertThat(p.maxAttempts()).isEqualTo(5);
		assertThat(p.backoffMs()).isEqualTo(250);
	}
}
