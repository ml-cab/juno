package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.ServeScheduleOptions;

class ContinuousLoraPolicyTest {

	@Test
	void allows_global_play_without_per_request_loras() {
		assertThat(ContinuousLoraPolicy.forbidden(false, ServeScheduleOptions.parse("continuous"))).isFalse();
	}

	@Test
	void rejects_per_request_loras_under_continuous() {
		assertThat(ContinuousLoraPolicy.forbidden(true, ServeScheduleOptions.parse("continuous"))).isTrue();
	}

	@Test
	void static_does_not_fail_closed() {
		assertThat(ContinuousLoraPolicy.forbidden(true, ServeScheduleOptions.defaults())).isFalse();
	}
}
