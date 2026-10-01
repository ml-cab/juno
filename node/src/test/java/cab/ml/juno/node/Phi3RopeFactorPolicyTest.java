package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.assertThatNoException;

import org.junit.jupiter.api.Test;

/**
 * Factor selection for Phi-3 LongRoPE: a model that carries both factor sets
 * rotates with the short ones, and a position at or beyond the original
 * training context fails closed instead of rotating with factors the sequence
 * was not started on.
 */
class Phi3RopeFactorPolicyTest {

	private static final float[] SHORT = { 1.0f, 1.1f };
	private static final float[] LONG = { 2.0f, 3.0f };

	/** Phi-3.5-mini's shape: trained to 131072, original context 4096, both sets present. */
	private static Phi3RopeConfig longRopeModel() {
		return new Phi3RopeConfig(10000f, 1.0f, 1.19f, 4096, 131072, SHORT, LONG);
	}

	@Test
	void longRopeModelSelectsShortFactors() {
		assertThat(longRopeModel().selectFactors()).isSameAs(SHORT);
	}

	@Test
	void longRopeModelAcceptsEveryPositionBelowOriginalContext() {
		Phi3RopeConfig cfg = longRopeModel();
		assertThatNoException().isThrownBy(() -> cfg.requirePosition(0));
		assertThatNoException().isThrownBy(() -> cfg.requirePosition(4095));
	}

	@Test
	void longRopeModelFailsClosedAtOriginalContext() {
		Phi3RopeConfig cfg = longRopeModel();
		assertThatThrownBy(() -> cfg.requirePosition(4096)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("4096");
		assertThatThrownBy(() -> Phi3Rope.ropeExt(new float[4], 4096, 1, 4, cfg))
				.isInstanceOf(IllegalStateException.class);
		assertThatThrownBy(() -> Phi3Rope.ropeExtBackward(new float[4], 5000, 1, 4, cfg))
				.isInstanceOf(IllegalStateException.class);
	}

	@Test
	void modelWithOnlyLongFactorsKeepsThemAndHasNoCap() {
		Phi3RopeConfig cfg = new Phi3RopeConfig(10000f, 1.0f, 1.0f, 4096, 131072, null, LONG);
		assertThat(cfg.selectFactors()).isSameAs(LONG);
		assertThatNoException().isThrownBy(() -> cfg.requirePosition(8000));
	}

	@Test
	void modelTrainedAtItsOriginalContextIsUnchanged() {
		Phi3RopeConfig cfg = new Phi3RopeConfig(10000f, 1.0f, 1.0f, 4096, 4096, SHORT, LONG);
		assertThat(cfg.selectFactors()).isSameAs(SHORT);
		assertThatNoException().isThrownBy(() -> cfg.requirePosition(8000));
	}
}
