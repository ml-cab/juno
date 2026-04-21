package cab.ml.juno.health;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;

import org.junit.jupiter.api.Test;

class NodeHealthTest {

	private NodeHealth health(double pressure) {
		return new NodeHealth("n1", "node", pressure, 1_000_000L, 10_000_000L, 0.5, 50.0, -1.0, Instant.now());
	}

	@Test
	void vram_warning_detected_at_threshold() {
		assertThat(health(0.90).isVramWarning(0.90)).isTrue();
		assertThat(health(0.89).isVramWarning(0.90)).isFalse();
	}

	@Test
	void vram_critical_detected_at_threshold() {
		assertThat(health(0.98).isVramCritical(0.98)).isTrue();
		assertThat(health(0.97).isVramCritical(0.98)).isFalse();
	}

	@Test
	void age_millis_is_near_zero_for_fresh_snapshot() {
		assertThat(health(0.5).ageMillis()).isLessThan(100);
	}

	@Test
	void rejects_pressure_out_of_range() {
		assertThatThrownBy(() -> new NodeHealth("n1", "node", 1.1, 0, 1, 0.5, 50, -1.0, Instant.now()))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new NodeHealth("n1", "node", -0.1, 0, 1, 0.5, 50, -1.0, Instant.now()))
				.isInstanceOf(IllegalArgumentException.class);
	}
}