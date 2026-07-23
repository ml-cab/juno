package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("DoraMagnitude")
class DoraMagnitudeTest {

	@Test
	@DisplayName("rejects non-positive length and non-finite values")
	void validation() {
		assertThatThrownBy(() -> new DoraMagnitude(0)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> DoraMagnitude.fromValues(new float[] { 1f, Float.NaN }))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("fromValues copies and zeroGrad clears")
	void copy_and_zero() {
		DoraMagnitude m = DoraMagnitude.fromValues(new float[] { 1f, 2f, 3f });
		assertThat(m.length()).isEqualTo(3);
		assertThat(m.values()).containsExactly(1f, 2f, 3f);
		m.grad()[1] = 9f;
		m.zeroGrad();
		assertThat(m.grad()).containsOnly(0f);
	}

	@Test
	@DisplayName("copyFrom requires matching length")
	void copy_from() {
		DoraMagnitude a = DoraMagnitude.fromValues(new float[] { 1f, 2f });
		DoraMagnitude b = new DoraMagnitude(2);
		b.copyFrom(a);
		assertThat(b.values()).containsExactly(1f, 2f);
		assertThatThrownBy(() -> b.copyFrom(new DoraMagnitude(3))).isInstanceOf(IllegalArgumentException.class);
	}
}
