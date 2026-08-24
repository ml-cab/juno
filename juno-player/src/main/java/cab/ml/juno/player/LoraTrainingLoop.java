/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.player;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraGradients;
import cab.ml.juno.lora.LoraLearningRateSchedule;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraTrainContext;
import cab.ml.juno.node.LoraGradientBatch;
import cab.ml.juno.node.LoraGradientResult;
import cab.ml.juno.node.LoraMetricsIdentity;
import cab.ml.juno.node.LoraNormRefreshEvent;
import cab.ml.juno.node.LoraTrainEvent;
import cab.ml.juno.node.LoraTrainableHandler;
import cab.ml.juno.node.LoraValidationEvent;

/**
 * Shared LoRA training orchestration: accumulation, schedules, dropout,
 * validation early stopping, and best-weight restoration.
 */
public final class LoraTrainingLoop {

	public enum StopReason {
		TARGET_REACHED, PATIENCE_EXHAUSTED, LOW_LOSS_GUARD, MAX_ITERATIONS, NO_DATA
	}

	/**
	 * Rich training outcome. Legacy callers can adapt via
	 * {@link #toLegacy(TrainingResult)}.
	 */
	public record TrainingResult(float finalTrainLoss, float finalValidationLoss, float bestValidationLoss,
			int passCount, int bestPass, int optimizerUpdateCount, StopReason stopReason, boolean validationDisabled,
			String validationWarning) {

		public boolean targetReached() {
			return stopReason == StopReason.TARGET_REACHED;
		}
	}

	@FunctionalInterface
	public interface GradientComputer {
		LoraGradientResult compute(int[] tokens, boolean[] lossMask, LoraTrainContext ctx);
	}

	@FunctionalInterface
	public interface LossEvaluator {
		LoraGradientResult evaluate(int[] tokens, boolean[] lossMask);
	}

	/**
	 * Invoked after each training pass (1-based). {@code validationLoss} is NaN when
	 * validation is disabled for this run.
	 */
	@FunctionalInterface
	public interface PassListener {
		void onPass(int pass, float trainLoss, float validationLoss, int optimizerUpdates);
	}

	public record TrainUnit(int[] tokens, boolean[] lossMask) {
	}

	private LoraTrainingLoop() {
	}

	public static LoraTrainer.TrainUntilResult toLegacy(TrainingResult r) {
		return new LoraTrainer.TrainUntilResult(r.finalTrainLoss(), r.passCount(), r.targetReached());
	}

	/**
	 * Deterministic Fisher–Yates split of unit indices. Validation count is clamped
	 * to {@code [1, n-1]} when validation is enabled and {@code n >= 2}.
	 */
	public static SplitResult splitUnits(int unitCount, float validationSplit, long seed) {
		if (unitCount < 0)
			throw new IllegalArgumentException("unitCount must be >= 0");
		List<Integer> order = new ArrayList<>(unitCount);
		for (int i = 0; i < unitCount; i++)
			order.add(i);
		Collections.shuffle(order, new Random(seed));
		if (validationSplit <= 0f || unitCount < 2) {
			String warn = unitCount < 2 && validationSplit > 0f
					? "validation disabled: fewer than two units"
					: null;
			return new SplitResult(List.copyOf(order), List.of(), warn != null, warn);
		}
		int valCount = Math.round(unitCount * validationSplit);
		valCount = Math.max(1, Math.min(unitCount - 1, valCount));
		List<Integer> val = List.copyOf(order.subList(0, valCount));
		List<Integer> train = List.copyOf(order.subList(valCount, unitCount));
		return new SplitResult(train, val, false, null);
	}

	public record SplitResult(List<Integer> trainIndices, List<Integer> validationIndices, boolean validationDisabled,
			String warning) {
	}

	public static int plannedUpdates(int trainChunkCount, int accumulationSteps, int maxPasses) {
		if (trainChunkCount < 1 || maxPasses < 1)
			return 0;
		int perPass = (trainChunkCount + accumulationSteps - 1) / accumulationSteps;
		return perPass * maxPasses;
	}

	public static LoraLearningRateSchedule buildSchedule(LoraTrainingConfig config, int plannedOptimizerUpdates) {
		if (config.lrSchedule() == LoraLearningRateSchedule.Mode.CONSTANT)
			return LoraLearningRateSchedule.constant(config.learningRate());
		int total = Math.max(plannedOptimizerUpdates, Math.max(1, config.warmupUpdates()));
		return LoraLearningRateSchedule.warmupCosine(config.learningRate(), config.minLearningRate(),
				config.warmupUpdates(), total);
	}

	public static TrainingResult train(List<TrainUnit> units, LoraTrainingConfig config, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, GradientComputer gradients, LossEvaluator evaluator, float lossTarget,
			int maxPasses, float earlyStopGuard) {
		return train(units, config, adapters, optimizer, gradients, evaluator, lossTarget, maxPasses, earlyStopGuard,
				config.chunkTokens(), null);
	}

	public static TrainingResult train(List<TrainUnit> units, LoraTrainingConfig config, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, GradientComputer gradients, LossEvaluator evaluator, float lossTarget,
			int maxPasses, float earlyStopGuard, int chunkTokens) {
		return train(units, config, adapters, optimizer, gradients, evaluator, lossTarget, maxPasses, earlyStopGuard,
				chunkTokens, null);
	}

	public static TrainingResult train(List<TrainUnit> units, LoraTrainingConfig config, LoraAdapterSet adapters,
			LoraAdamOptimizer optimizer, GradientComputer gradients, LossEvaluator evaluator, float lossTarget,
			int maxPasses, float earlyStopGuard, int chunkTokens, PassListener passListener) {
		if (units == null || units.isEmpty())
			return new TrainingResult(Float.NaN, Float.NaN, Float.NaN, 0, -1, 0, StopReason.NO_DATA, true, "no units");

		SplitResult split = splitUnits(units.size(), config.validationSplit(), config.seed());
		List<TrainUnit> trainUnits = select(units, split.trainIndices());
		List<TrainUnit> valUnits = select(units, split.validationIndices());
		boolean validationOn = !split.validationDisabled() && !valUnits.isEmpty() && config.validationPatience() > 0;

		List<LoraTrainingSequences.MaskedChunk> trainChunks = flattenChunks(trainUnits, chunkTokens);
		int planned = plannedUpdates(trainChunks.size(), config.gradientAccumulationSteps(), maxPasses);
		LoraLearningRateSchedule schedule = buildSchedule(config, planned);

		WeightSnapshot bestWeights = null;
		float bestVal = Float.POSITIVE_INFINITY;
		int bestPass = -1;
		int badChecks = 0;
		float lastTrain = Float.NaN;
		float lastVal = Float.NaN;
		StopReason stop = StopReason.MAX_ITERATIONS;
		int pass = 0;

		for (; pass < maxPasses; pass++) {
			lastTrain = runTrainPass(trainChunks, config, adapters, optimizer, gradients, schedule);
			if (validationOn) {
				long t0 = System.currentTimeMillis();
				lastVal = evaluateUnits(valUnits, evaluator, chunkTokens);
				boolean best = lastVal + config.validationMinDelta() < bestVal;
				commitValidation(config.metricsIdentity(), lastVal, 0, System.currentTimeMillis() - t0, best,
						pass + 1, optimizer.step(), lastTrain);
				if (best) {
					bestVal = lastVal;
					bestPass = pass;
					badChecks = 0;
					if (config.restoreBest())
						bestWeights = snapshotAll(adapters);
				} else {
					badChecks++;
				}
			}
			if (passListener != null)
				passListener.onPass(pass + 1, lastTrain, validationOn ? lastVal : Float.NaN, optimizer.step());

			if (Float.isFinite(lastTrain) && lastTrain < earlyStopGuard) {
				stop = StopReason.LOW_LOSS_GUARD;
				pass++;
				break;
			}
			if (Float.isFinite(lastTrain) && lastTrain <= lossTarget) {
				stop = StopReason.TARGET_REACHED;
				pass++;
				break;
			}
			if (validationOn && badChecks >= config.validationPatience()) {
				stop = StopReason.PATIENCE_EXHAUSTED;
				pass++;
				break;
			}
		}

		int updateCount = optimizer.step();
		if (config.restoreBest() && bestWeights != null) {
			restoreAll(adapters, bestWeights);
			optimizer.reset();
		}

		return new TrainingResult(lastTrain, lastVal, bestVal == Float.POSITIVE_INFINITY ? Float.NaN : bestVal, pass,
				bestPass, updateCount, stop, !validationOn, split.warning());
	}

	/** One accumulation pass over pre-chunked training data (no validation). */
	public static float runTrainPass(List<LoraTrainingSequences.MaskedChunk> chunks, LoraTrainingConfig config,
			LoraAdapterSet adapters, LoraAdamOptimizer optimizer, GradientComputer gradients,
			LoraLearningRateSchedule schedule) {
		if (chunks.isEmpty())
			return Float.NaN;
		int accum = config.gradientAccumulationSteps();
		LoraGradientBatch batch = new LoraGradientBatch();
		float lastMean = Float.NaN;
		int predSum = 0;
		int globalChunk = 0;

		adapters.zeroAllGrads();
		for (int i = 0; i < chunks.size(); i++) {
			var chunk = chunks.get(i);
			int nextUpdate = optimizer.step() + 1;
			LoraTrainContext ctx = config.dropout() > 0f
					? new LoraTrainContext(config.seed(), config.dropout(), nextUpdate, globalChunk)
					: LoraTrainContext.disabled();
			LoraGradientResult r = gradients.compute(chunk.tokens(), chunk.lossMask(), ctx);
			batch.add(r);
			predSum += r.predictionCount();
			globalChunk++;

			boolean groupFull = batch.chunkCount() >= accum;
			boolean lastChunk = i == chunks.size() - 1;
			if (groupFull || lastChunk) {
				if (batch.predictionCount() > 0)
					lastMean = stepOptimizer(adapters, optimizer, batch, predSum, config, schedule);
				batch.clear();
				predSum = 0;
				if (!lastChunk)
					adapters.zeroAllGrads();
			}
		}
		return lastMean;
	}

	public static float evaluateUnits(List<TrainUnit> units, LossEvaluator evaluator) {
		return evaluateUnits(units, evaluator, 32);
	}

	public static float evaluateUnits(List<TrainUnit> units, LossEvaluator evaluator, int chunkTokens) {
		float lossSum = 0f;
		int preds = 0;
		for (TrainUnit u : units) {
			for (var chunk : LoraTrainingSequences.chunk(u.tokens(), u.lossMask(), chunkTokens)) {
				LoraGradientResult r = evaluator.evaluate(chunk.tokens(), chunk.lossMask());
				lossSum += r.lossSum();
				preds += r.predictionCount();
			}
		}
		return preds == 0 ? Float.NaN : lossSum / preds;
	}

	private static float stepOptimizer(LoraAdapterSet adapters, LoraAdamOptimizer optimizer, LoraGradientBatch batch,
			int numTokens, LoraTrainingConfig config, LoraLearningRateSchedule schedule) {
		LoraMetricsIdentity identity = config.metricsIdentity();
		LoraTrainEvent event = new LoraTrainEvent();
		event.begin();
		identity.apply(event);
		event.step = optimizer.step() + 1;
		event.numTokens = numTokens;
		event.chunkCount = batch.chunkCount();
		event.predictionCount = batch.predictionCount();
		event.forwardMs = batch.forwardMs();
		event.backwardMs = batch.backwardMs();
		batch.timing().apply(event);
		event.dropout = config.dropout();
		event.loraPlusRatio = (float) config.loraPlusRatio();

		LoraGradients.PrepResult prep = LoraGradients.prepare(adapters, batch.predictionCount(), config.maxGradNorm());
		event.globalGradNorm = (float) prep.globalNorm();
		event.clipScale = prep.scale();
		event.clipped = prep.clipped();

		double lr = schedule.learningRate(event.step);
		long t0 = System.currentTimeMillis();
		optimizer.step(adapters, lr);
		event.optimizerMs = System.currentTimeMillis() - t0;
		event.learningRateA = (float) optimizer.lastLearningRateA();
		event.learningRateB = (float) optimizer.lastLearningRateB();

		float mean = batch.meanLoss();
		event.loss = mean;
		event.totalMs = event.forwardMs + event.backwardMs + event.optimizerMs;
		event.commit();

		if (config.mode() == LoraMode.DORA)
			commitNormRefresh(identity, adapters, "post-step", 0L);

		return mean;
	}

	private static List<TrainUnit> select(List<TrainUnit> units, List<Integer> indices) {
		List<TrainUnit> out = new ArrayList<>(indices.size());
		for (int i : indices)
			out.add(units.get(i));
		return out;
	}

	private static List<LoraTrainingSequences.MaskedChunk> flattenChunks(List<TrainUnit> units, int chunkTokens) {
		List<LoraTrainingSequences.MaskedChunk> out = new ArrayList<>();
		for (TrainUnit u : units)
			out.addAll(LoraTrainingSequences.chunk(u.tokens(), u.lossMask(), chunkTokens));
		return out;
	}

	private record WeightSnapshot(Map<LoraAdapter, float[][]> dense,
			Map<cab.ml.juno.lora.QaLoraAdapter, float[][]> qa) {
	}

	private static WeightSnapshot snapshotAll(LoraAdapterSet adapters) {
		return new WeightSnapshot(snapshot(adapters), snapshotQa(adapters));
	}

	private static void restoreAll(LoraAdapterSet adapters, WeightSnapshot snap) {
		restore(adapters, snap.dense());
		restoreQa(adapters, snap.qa());
	}

	private static Map<LoraAdapter, float[][]> snapshot(LoraAdapterSet adapters) {
		Map<LoraAdapter, float[][]> snap = new IdentityHashMap<>();
		for (LoraAdapter a : adapters.all())
			snap.put(a, new float[][] { a.a().clone(), a.b().clone() });
		return snap;
	}

	private static Map<cab.ml.juno.lora.QaLoraAdapter, float[][]> snapshotQa(LoraAdapterSet adapters) {
		Map<cab.ml.juno.lora.QaLoraAdapter, float[][]> snap = new IdentityHashMap<>();
		for (cab.ml.juno.lora.QaLoraAdapter a : adapters.allQa())
			snap.put(a, new float[][] { a.a().clone(), a.b().clone() });
		return snap;
	}

	private static void restore(LoraAdapterSet adapters, Map<LoraAdapter, float[][]> snap) {
		for (LoraAdapter a : adapters.all()) {
			float[][] w = snap.get(a);
			if (w == null)
				continue;
			System.arraycopy(w[0], 0, a.a(), 0, a.a().length);
			System.arraycopy(w[1], 0, a.b(), 0, a.b().length);
			a.zeroGrad();
		}
	}

	private static void restoreQa(LoraAdapterSet adapters, Map<cab.ml.juno.lora.QaLoraAdapter, float[][]> snap) {
		for (cab.ml.juno.lora.QaLoraAdapter a : adapters.allQa()) {
			float[][] w = snap.get(a);
			if (w == null)
				continue;
			System.arraycopy(w[0], 0, a.a(), 0, a.a().length);
			System.arraycopy(w[1], 0, a.b(), 0, a.b().length);
			a.zeroGrad();
		}
	}

	/** Emit a validation JFR event (optional observability). */
	public static void commitValidation(float loss, int predictions, long durationMs, boolean bestSoFar) {
		commitValidation(null, loss, predictions, durationMs, bestSoFar, 0, 0, Float.NaN);
	}

	public static void commitValidation(LoraMetricsIdentity identity, float loss, int predictions, long durationMs,
			boolean bestSoFar, int passIndex, int optimizerStep, float trainLossAtEval) {
		LoraValidationEvent ev = new LoraValidationEvent();
		ev.begin();
		if (identity != null)
			identity.apply(ev);
		ev.loss = loss;
		ev.predictionCount = predictions;
		ev.durationMs = durationMs;
		ev.bestSoFar = bestSoFar;
		ev.passIndex = passIndex;
		ev.optimizerStep = optimizerStep;
		ev.trainLossAtEval = trainLossAtEval;
		ev.commit();
	}

	/** Emit a DoRA norm-refresh JFR event; no-op when there are no magnitudes. */
	public static void commitNormRefresh(LoraMetricsIdentity identity, LoraAdapterSet adapters, String reason,
			long durationMs) {
		int projections = adapters != null ? adapters.magnitudes().size() : 0;
		if (projections == 0)
			return;
		int layers = 0;
		if (adapters != null) {
			java.util.HashSet<Integer> layerSet = new java.util.HashSet<>();
			for (String key : adapters.magnitudes().keySet())
				layerSet.add(LoraAdapterSet.keyLayer(key));
			layers = layerSet.size();
		}
		LoraNormRefreshEvent ev = new LoraNormRefreshEvent();
		ev.begin();
		if (identity != null)
			identity.apply(ev);
		ev.layerCount = layers;
		ev.projectionCount = projections;
		ev.durationMs = durationMs;
		ev.reason = reason != null ? reason : "";
		ev.commit();
	}
}
