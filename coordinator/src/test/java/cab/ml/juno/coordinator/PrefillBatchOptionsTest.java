package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class PrefillBatchOptionsTest {

	@Test
	void defaults_to_32() {
		assertThat(PrefillBatchOptions.defaults().chunkSize()).isEqualTo(32);
	}

	@Test
	void resolve_cli_overrides_default() {
		assertThat(PrefillBatchOptions.resolve(64).chunkSize()).isEqualTo(64);
	}

	@Test
	void of_rejects_zero() {
		assertThatThrownBy(() -> PrefillBatchOptions.of(0)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining(">= 1");
	}

	@Test
	void chunk_size_1_is_valid() {
		assertThat(PrefillBatchOptions.of(1).chunkSize()).isEqualTo(1);
	}
}
