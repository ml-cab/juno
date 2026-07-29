package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraLearningRateSchedule;
import cab.ml.juno.node.LoraGradientResult;

@DisplayName("LoraTrainingLoop")
class LoraTrainingLoopTest {

	@Test
	@DisplayName("deterministic split is disjoint and reproducible")
	void deterministic_split_disjoint() {
		LoraTrainingLoop.SplitResult a = LoraTrainingLoop.splitUnits(10, 0.3f, 42L);
		LoraTrainingLoop.SplitResult b = LoraTrainingLoop.splitUnits(10, 0.3f, 42L);
		assertThat(a.trainIndices()).isEqualTo(b.trainIndices());
		assertThat(a.validationIndices()).isEqualTo(b.validationIndices());
		Set<Integer> train = new HashSet<>(a.trainIndices());
		Set<Integer> val = new HashSet<>(a.validationIndices());
		assertThat(train).doesNotContainAnyElementsOf(val);
		assertThat(train.size() + val.size()).isEqualTo(10);
		assertThat(val).hasSizeBetween(1, 9);
	}

	@Test
	@DisplayName("one-unit fallback disables validation with a warning")
	void one_unit_fallback() {
		LoraTrainingLoop.SplitResult s = LoraTrainingLoop.splitUnits(1, 0.5f, 1L);
		assertThat(s.validationDisabled()).isTrue();
		assertThat(s.validationIndices()).isEmpty();
		assertThat(s.warning()).contains("fewer than two");
	}

	@Test
	@DisplayName("weighted validation aggregates token counts")
	void weighted_aggregation() {
		AtomicInteger calls = new AtomicInteger();
		float loss = LoraTrainingLoop.evaluateUnits(List.of(unit(new int[] { 1, 2, 3 }, new boolean[] { true, true }),
				unit(new int[] { 4, 5 }, new boolean[] { true })), (tokens, mask) -> {
					calls.incrementAndGet();
					int n = 0;
					for (boolean m : mask)
						if (m)
							n++;
					return new LoraGradientResult(2f * n, n, 1L, 0L);
				}, 32);
		assertThat(loss).isEqualTo(2f);
		assertThat(calls.get()).isGreaterThanOrEqualTo(2);
	}

	@Test
	@DisplayName("patience and min-delta stop with restore and optimizer reset")
	void patience_restore_reset() {
		LoraAdapterSet adapters = new LoraAdapterSet();
		LoraAdapter adapter = new LoraAdapter(2, 4, 4, 2f, new java.util.Random(1));
		for (int i = 0; i < adapter.b().length; i++)
			adapter.b()[i] = 0.5f;
		adapters.add(0, "wq", adapter);
		float[] bestB = adapter.b().clone();

		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-2, 0.9, 0.999, 1e-8, 0.0);
		AtomicInteger trainCalls = new AtomicInteger();
		AtomicInteger valCalls = new AtomicInteger();

		LoraTrainingConfig config = LoraTrainingConfig.builder().learningRate(1e-2).weightDecay(0).validationSplit(0.5f)
				.validationPatience(2).validationMinDelta(0.01f).restoreBest(true).seed(7).maxGradNorm(0f).build();

		List<LoraTrainingLoop.TrainUnit> units = List.of(unit(new int[] { 1, 2, 3 }, new boolean[] { true, true }),
				unit(new int[] { 4, 5, 6 }, new boolean[] { true, true }),
				unit(new int[] { 7, 8, 9 }, new boolean[] { true, true }),
				unit(new int[] { 10, 11, 12 }, new boolean[] { true, true }));

		LoraTrainingLoop.TrainingResult result = LoraTrainingLoop.train(units, config, adapters, opt, (tokens, mask,
				ctx) -> {
			trainCalls.incrementAndGet();
			for (int i = 0; i < adapter.gradA().length; i++)
				adapter.gradA()[i] = 0.01f;
			return new LoraGradientResult(1.0f, 1, 1L, 1L);
		}, (tokens, mask) -> {
			int n = valCalls.incrementAndGet();
			// Never improves after first check.
			float loss = n == 1 ? 1.0f : 1.5f;
			return new LoraGradientResult(loss, 1, 1L, 0L);
		}, 0.01f, 20, 0.0f);

		assertThat(result.stopReason()).isEqualTo(LoraTrainingLoop.StopReason.PATIENCE_EXHAUSTED);
		assertThat(opt.step()).isEqualTo(0); // reset after restore
		assertThat(adapter.b()).containsExactly(bestB);
		assertThat(result.bestPass()).isEqualTo(0);
	}

	@Test
	@DisplayName("schedule update count matches optimizer steps after accumulation")
	void schedule_update_count() {
		assertThat(LoraTrainingLoop.plannedUpdates(5, 2, 3)).isEqualTo(9);
		LoraTrainingConfig config = LoraTrainingConfig.builder().learningRate(1e-3)
				.lrSchedule(LoraLearningRateSchedule.Mode.COSINE).minLearningRate(1e-5).warmupUpdates(2).build();
		LoraLearningRateSchedule schedule = LoraTrainingLoop.buildSchedule(config, 10);
		assertThat(schedule.mode()).isEqualTo(LoraLearningRateSchedule.Mode.COSINE);
		assertThat(schedule.totalUpdates()).isEqualTo(10);
	}

	@Test
	@DisplayName("distinct stop reasons for target and max iterations")
	void distinct_stop_reasons() {
		LoraAdapterSet adapters = new LoraAdapterSet();
		adapters.add(0, "wq", new LoraAdapter(2, 4, 4, 2f, new java.util.Random(2)));
		LoraAdamOptimizer opt = LoraAdamOptimizer.defaults(1e-3);
		LoraTrainingConfig config = LoraTrainingConfig.builder().learningRate(1e-3).weightDecay(0).maxGradNorm(0f)
				.build();
		List<LoraTrainingLoop.TrainUnit> units = List.of(unit(new int[] { 1, 2 }, new boolean[] { true }));

		LoraTrainingLoop.TrainingResult hit = LoraTrainingLoop.train(units, config, adapters, opt,
				(t, m, c) -> new LoraGradientResult(0.5f, 1, 1L, 1L), (t, m) -> new LoraGradientResult(0.5f, 1, 1L, 0L),
				1.0f, 5, 0.0f);
		assertThat(hit.stopReason()).isEqualTo(LoraTrainingLoop.StopReason.TARGET_REACHED);

		opt.reset();
		LoraTrainingLoop.TrainingResult max = LoraTrainingLoop.train(units, config, adapters, opt,
				(t, m, c) -> new LoraGradientResult(2.0f, 1, 1L, 1L), (t, m) -> new LoraGradientResult(2.0f, 1, 1L, 0L),
				0.1f, 3, 0.0f);
		assertThat(max.stopReason()).isEqualTo(LoraTrainingLoop.StopReason.MAX_ITERATIONS);
		assertThat(max.passCount()).isEqualTo(3);
	}

	@Test
	@DisplayName("train and validation contexts are disjoint by unit index")
	void train_validation_disjointness() {
		LoraTrainingLoop.SplitResult split = LoraTrainingLoop.splitUnits(8, 0.25f, 99L);
		List<Integer> all = new ArrayList<>();
		all.addAll(split.trainIndices());
		all.addAll(split.validationIndices());
		assertThat(new HashSet<>(all)).hasSize(8);
		assertThat(split.trainIndices()).doesNotContainAnyElementsOf(split.validationIndices());
	}

	@Test
	@DisplayName("dropout context uses upcoming optimizer update index")
	void dropout_context_update_index() {
		LoraAdapterSet adapters = new LoraAdapterSet();
		adapters.add(0, "wq", new LoraAdapter(2, 4, 4, 2f, new java.util.Random(3)));
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-3, 0.9, 0.999, 1e-8, 0);
		LoraTrainingConfig config = LoraTrainingConfig.builder().learningRate(1e-3).dropout(0.1f).weightDecay(0)
				.maxGradNorm(0f).seed(5).build();
		List<LoraTrainingSequences.MaskedChunk> chunks = List
				.of(new LoraTrainingSequences.MaskedChunk(new int[] { 1, 2 }, new boolean[] { true }));
		List<Integer> updates = new ArrayList<>();
		LoraTrainingLoop.runTrainPass(chunks, config, adapters, opt, (tokens, mask, ctx) -> {
			updates.add(ctx.optimizerUpdate());
			assertThat(ctx.dropoutEnabled()).isTrue();
			return new LoraGradientResult(1f, 1, 1L, 1L);
		}, LoraLearningRateSchedule.constant(1e-3));
		assertThat(updates).containsExactly(1);
	}

	@Test
	@DisplayName("pass listener receives one callback per completed pass")
	void pass_listener_invoked() {
		LoraAdapterSet adapters = new LoraAdapterSet();
		adapters.add(0, "wq", new LoraAdapter(2, 4, 4, 2f, new java.util.Random(4)));
		LoraAdamOptimizer opt = new LoraAdamOptimizer(1e-3, 0.9, 0.999, 1e-8, 0);
		LoraTrainingConfig config = LoraTrainingConfig.builder().learningRate(1e-3).weightDecay(0).maxGradNorm(0f)
				.build();
		List<LoraTrainingLoop.TrainUnit> units = List.of(unit(new int[] { 1, 2 }, new boolean[] { true }));
		List<Integer> passes = new ArrayList<>();
		LoraTrainingLoop.train(units, config, adapters, opt, (t, m, c) -> new LoraGradientResult(2.0f, 1, 1L, 1L),
				(t, m) -> new LoraGradientResult(2.0f, 1, 1L, 0L), 0.1f, 3, 0.0f, 32,
				(pass, trainLoss, valLoss, updates) -> passes.add(pass));
		assertThat(passes).containsExactly(1, 2, 3);
	}

	private static LoraTrainingLoop.TrainUnit unit(int[] tokens, boolean[] mask) {
		return new LoraTrainingLoop.TrainUnit(tokens, mask);
	}
}
