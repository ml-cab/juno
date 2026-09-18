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
	void static_also_fails_closed_pending_per_request_wiring() {
		assertThat(ContinuousLoraPolicy.forbidden(true, ServeScheduleOptions.defaults())).isTrue();
	}

	@Test
	void schedule_null_still_fails_closed_when_per_request_loras_present() {
		assertThat(ContinuousLoraPolicy.forbidden(true, null)).isTrue();
	}
}
