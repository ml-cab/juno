package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraStepTiming")
class LoraStepTimingTest {

	@Test
	@DisplayName("aggregates across LoraGradientBatch")
	void batch_aggregates_timing() {
		LoraStepTiming a = new LoraStepTiming();
		a.frozenForwardMs = 10;
		a.frozenTransposeBackwardMs = 20;
		a.adapterBackwardMs = 3;
		a.attentionNonlinearMs = 5;
		LoraStepTiming b = new LoraStepTiming();
		b.frozenForwardMs = 1;
		b.frozenTransposeBackwardMs = 2;
		b.adapterBackwardMs = 4;

		LoraGradientBatch batch = new LoraGradientBatch();
		batch.add(new LoraGradientResult(1f, 1, 16L, 28L, a));
		batch.add(new LoraGradientResult(1f, 1, 8L, 10L, b));

		assertThat(batch.timing().frozenForwardMs).isEqualTo(11);
		assertThat(batch.timing().frozenTransposeBackwardMs).isEqualTo(22);
		assertThat(batch.timing().adapterBackwardMs).isEqualTo(7);
		assertThat(batch.timing().attentionNonlinearMs).isEqualTo(5);

		LoraTrainEvent ev = new LoraTrainEvent();
		batch.timing().apply(ev);
		assertThat(ev.frozenForwardMs).isEqualTo(11);
		assertThat(ev.frozenTransposeBackwardMs).isEqualTo(22);
		assertThat(ev.adapterBackwardMs).isEqualTo(7);
	}

	@Test
	@DisplayName("4-arg LoraGradientResult keeps zero timing")
	void compat_ctor_zero_timing() {
		LoraGradientResult r = new LoraGradientResult(1f, 2, 3L, 4L);
		assertThat(r.timing().frozenForwardMs).isZero();
		assertThat(r.timing().frozenTransposeBackwardMs).isZero();
	}
}
