package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.ServeScheduleOptions;

class ServeSchedulePolicyTest {

	private String prev;

	@AfterEach
	void restore() {
		if (prev == null)
			System.clearProperty(ServeScheduleOptions.ENV);
		else
			System.setProperty(ServeScheduleOptions.ENV, prev);
	}

	@Test
	void local_keeps_continuous() {
		var res = ServeSchedulePolicy.resolve(ServeScheduleOptions.parse("continuous"),
				ServeSchedulePolicy.Topology.LOCAL);
		assertThat(res.fallback()).isFalse();
		assertThat(res.options().mode()).isEqualTo(ServeScheduleOptions.Mode.CONTINUOUS);
		assertThat(res.options().usesPagedKv()).isTrue();
	}

	@Test
	void cluster_falls_back_to_static() {
		var res = ServeSchedulePolicy.resolve(ServeScheduleOptions.parse("continuous"),
				ServeSchedulePolicy.Topology.CLUSTER);
		assertThat(res.fallback()).isTrue();
		assertThat(res.warning()).isEqualTo(ServeSchedulePolicy.CLUSTER_FALLBACK_WARNING);
		assertThat(res.options().mode()).isEqualTo(ServeScheduleOptions.Mode.STATIC);
		assertThat(res.options().usesPagedKv()).isFalse();
	}

	@Test
	void cluster_static_unchanged() {
		var res = ServeSchedulePolicy.resolve(ServeScheduleOptions.defaults(),
				ServeSchedulePolicy.Topology.CLUSTER);
		assertThat(res.fallback()).isFalse();
		assertThat(res.options().mode()).isEqualTo(ServeScheduleOptions.Mode.STATIC);
	}

	@Test
	void applyToEnv_writes_fallback() {
		prev = System.getProperty(ServeScheduleOptions.ENV);
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		var res = ServeSchedulePolicy.resolve(ServeScheduleOptions.parse("continuous"),
				ServeSchedulePolicy.Topology.CLUSTER);
		res.applyToEnv();
		assertThat(ServeScheduleOptions.fromEnv().mode()).isEqualTo(ServeScheduleOptions.Mode.STATIC);
	}

	@Test
	void running_set_upgrades_disabled_batch() {
		assertThat(ServeSchedulePolicy.runningSetConfig(BatchConfig.disabled()).maxBatchSize())
				.isEqualTo(ServeSchedulePolicy.DEFAULT_RUNNING_SET);
		assertThat(ServeSchedulePolicy.runningSetConfig(BatchConfig.of(4, 20)).maxBatchSize()).isEqualTo(4);
	}
}
