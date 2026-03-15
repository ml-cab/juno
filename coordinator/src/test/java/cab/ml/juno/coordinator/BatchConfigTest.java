package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.BatchConfig;

class BatchConfigTest {

	@Test
	void defaults_have_sensible_values() {
		BatchConfig cfg = BatchConfig.defaults();
		assertThat(cfg.maxBatchSize()).isEqualTo(8);
		assertThat(cfg.batchWindowMs()).isEqualTo(50);
	}

	@Test
	void disabled_has_batch_size_one_and_zero_window() {
		BatchConfig cfg = BatchConfig.disabled();
		assertThat(cfg.maxBatchSize()).isEqualTo(1);
		assertThat(cfg.batchWindowMs()).isEqualTo(0);
	}

	@Test
	void disabled_reports_batching_not_enabled() {
		assertThat(BatchConfig.disabled().isBatchingEnabled()).isFalse();
	}

	@Test
	void defaults_reports_batching_enabled() {
		assertThat(BatchConfig.defaults().isBatchingEnabled()).isTrue();
	}

	@Test
	void custom_config_round_trips_values() {
		BatchConfig cfg = BatchConfig.of(16, 100);
		assertThat(cfg.maxBatchSize()).isEqualTo(16);
		assertThat(cfg.batchWindowMs()).isEqualTo(100);
		assertThat(cfg.isBatchingEnabled()).isTrue();
	}

	@Test
	void rejects_zero_batch_size() {
		assertThatThrownBy(() -> BatchConfig.of(0, 50)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("maxBatchSize");
	}

	@Test
	void rejects_negative_window() {
		assertThatThrownBy(() -> BatchConfig.of(8, -1)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("batchWindowMs");
	}
}