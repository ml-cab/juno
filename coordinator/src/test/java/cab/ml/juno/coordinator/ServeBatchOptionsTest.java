package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ServeBatchOptions — parse and BatchConfig mapping")
class ServeBatchOptionsTest {

	@Test
	@DisplayName("defaults map to disabled batching")
	void defaults_disabled() {
		assertThat(ServeBatchOptions.defaults().toBatchConfig()).isEqualTo(BatchConfig.disabled());
	}

	@Test
	@DisplayName("parallel=1 disables batching regardless of window")
	void parallel_one_disabled() {
		BatchConfig cfg = ServeBatchOptions.of(1, 50).toBatchConfig();
		assertThat(cfg.isBatchingEnabled()).isFalse();
		assertThat(cfg.maxBatchSize()).isEqualTo(1);
		assertThat(cfg.batchWindowMs()).isZero();
	}

	@Test
	@DisplayName("parallel>1 enables batching with explicit window")
	void parallel_gt_one() {
		BatchConfig cfg = ServeBatchOptions.of(8, 100).toBatchConfig();
		assertThat(cfg.isBatchingEnabled()).isTrue();
		assertThat(cfg.maxBatchSize()).isEqualTo(8);
		assertThat(cfg.batchWindowMs()).isEqualTo(100);
	}

	@Test
	@DisplayName("parallel>1 without window defaults to 50ms")
	void parallel_default_window() {
		BatchConfig cfg = ServeBatchOptions.resolve(8, null).toBatchConfig();
		assertThat(cfg.batchWindowMs()).isEqualTo(50);
	}

	@Test
	@DisplayName("CLI overrides implicit defaults")
	void cli_overrides() {
		BatchConfig cfg = ServeBatchOptions.resolve(2, 10L).toBatchConfig();
		assertThat(cfg.maxBatchSize()).isEqualTo(2);
		assertThat(cfg.batchWindowMs()).isEqualTo(10);
	}

	@Test
	@DisplayName("invalid parallel fails closed")
	void invalid_parallel() {
		assertThatThrownBy(() -> ServeBatchOptions.of(0, 0)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining(">= 1");
	}

	@Test
	@DisplayName("negative window fails closed")
	void invalid_window() {
		assertThatThrownBy(() -> ServeBatchOptions.of(2, -1)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining(">= 0");
	}
}
