package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ServeScheduleOptions")
class ServeScheduleOptionsTest {

	private String prev;

	@AfterEach
	void restore() {
		if (prev == null)
			System.clearProperty(ServeScheduleOptions.ENV);
		else
			System.setProperty(ServeScheduleOptions.ENV, prev);
	}

	@Test
	@DisplayName("default schedule is static")
	void defaults_static() {
		prev = System.getProperty(ServeScheduleOptions.ENV);
		System.clearProperty(ServeScheduleOptions.ENV);
		assertThat(ServeScheduleOptions.fromEnv().mode()).isEqualTo(ServeScheduleOptions.Mode.STATIC);
		assertThat(ServeScheduleOptions.fromEnv().usesPagedKv()).isFalse();
	}

	@Test
	@DisplayName("continuous enables paged KV path")
	void continuous_paged() {
		prev = System.getProperty(ServeScheduleOptions.ENV);
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		ServeScheduleOptions opts = ServeScheduleOptions.fromEnv();
		assertThat(opts.mode()).isEqualTo(ServeScheduleOptions.Mode.CONTINUOUS);
		assertThat(opts.usesPagedKv()).isTrue();
	}

	@Test
	@DisplayName("reject unknown mode")
	void reject_unknown() {
		assertThatThrownBy(() -> ServeScheduleOptions.parse("batch"))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
