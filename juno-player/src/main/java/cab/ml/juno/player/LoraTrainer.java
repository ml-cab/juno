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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraLearningRateSchedule;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.node.DoraInitializer;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LoraInitializer;
import cab.ml.juno.node.LoraTrainingHandler;
import cab.ml.juno.node.LoraTrainingHandlerFactory;
import cab.ml.juno.node.QaLoraInitializer;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.tokenizer.GgufTokenizer;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * Programmatic LoRA fine-tuning on a single full-model shard (same layout as the
 * {@code ./juno lora} REPL).
 */
public final class LoraTrainer implements AutoCloseable {

	/** Outcome of {@link #trainRawTextUntil} / {@link #trainQaPairUntil}. */
	public record TrainUntilResult(float finalLoss, int iterations, boolean targetReached) {
	}

	private final LoraTrainingHandler handler;
	private final Tokenizer tokenizer;
	private final LoraAdamOptimizer optimizer;
	private final LoraAdapterSet adapters;
	private final Path adapterPath;
	private final Path modelPath;
	private final LoraTrainingConfig config;
	private final LlamaConfig modelConfig;

	private LoraTrainer(LoraTrainingHandler handler, Tokenizer tokenizer, LoraAdamOptimizer optimizer,
			LoraAdapterSet adapters, Path adapterPath, Path modelPath, LoraTrainingConfig config,
			LlamaConfig modelConfig) {
		this.handler = handler;
		this.tokenizer = tokenizer;
		this.optimizer = optimizer;
		this.adapters = adapters;
		this.adapterPath = adapterPath;
		this.modelPath = modelPath;
		this.config = config;
		this.modelConfig = modelConfig;
	}

	/**
	 * Legacy open: qv targets, accumulation 1, clipping disabled, legacy-normal init.
	 */
	public static LoraTrainer open(Path modelPath, Path adapterPath, int rank, float alpha, double lr)
			throws IOException {
		return open(modelPath, adapterPath, LoraTrainingConfig.builder().adapterConfig(LoraAdapterConfig.legacy(rank, alpha))
				.learningRate(lr).targets("qv").gradientAccumulationSteps(1).maxGradNorm(0f).build());
	}

	public static LoraTrainer open(Path modelPath, Path adapterPath, LoraTrainingConfig config) throws IOException {
		LlamaConfig cfg;
		Tokenizer tokenizer;
		LoraAdapterSet adapters;
		Path ap = adapterPath != null ? adapterPath : defaultAdapterPath(modelPath);
		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
			if (Files.exists(ap)) {
				adapters = LoraAdapterSet.load(ap);
				LoraInitializer.validate(adapters, cfg);
				DoraInitializer.verifyFingerprints(r, adapters);
				DoraInitializer.attachMissingDoraState(r, cfg, adapters);
				QaLoraInitializer.verifyFingerprints(r, adapters);
			} else if (config.mode() == LoraMode.DORA) {
				adapters = DoraInitializer.create(r, cfg, config.targets(), config.adapterConfig(),
						new Random(config.seed()));
			} else if (config.mode() == LoraMode.QA_LORA) {
				adapters = QaLoraInitializer.create(r, cfg, config.targets(), config.adapterConfig(),
						new Random(config.seed()), config.groupWidth(), config.mergeCapability());
			} else {
				adapters = LoraInitializer.create(cfg, config.targets(), config.adapterConfig(),
						new Random(config.seed()));
			}
		}

		ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, cfg.numLayers(), true, true);
		ShardContext ctx = ShardContext.from(assignment, cfg.vocabSize(), cfg.hiddenDim(), cfg.numHeads());
		LoraTrainingHandler handler = LoraTrainingHandlerFactory.create(modelPath, ctx, adapters);
		LoraAdamOptimizer optimizer = new LoraAdamOptimizer(config.learningRate(), 0.9, 0.999, 1e-8,
				config.weightDecay(), config.loraPlusRatio());
		LoraTrainingConfig enriched = enrichMetricsLabels(config, cfg.architecture());
		return new LoraTrainer(handler, tokenizer, optimizer, adapters, ap, modelPath, enriched, cfg);
	}

	/** Fill architecture for JFR identity when the caller left it blank. */
	static LoraTrainingConfig enrichMetricsLabels(LoraTrainingConfig config, String architecture) {
		if (config.architecture() != null && !config.architecture().isBlank())
			return config;
		return LoraTrainingConfig.builder().adapterConfig(config.adapterConfig()).targets(config.targets())
				.learningRate(config.learningRate()).gradientAccumulationSteps(config.gradientAccumulationSteps())
				.maxGradNorm(config.maxGradNorm()).lrSchedule(config.lrSchedule())
				.minLearningRate(config.minLearningRate()).warmupUpdates(config.warmupUpdates())
				.weightDecay(config.weightDecay()).loraPlusRatio(config.loraPlusRatio()).dropout(config.dropout())
				.seed(config.seed()).validationSplit(config.validationSplit())
				.validationPatience(config.validationPatience()).validationMinDelta(config.validationMinDelta())
				.restoreBest(config.restoreBest()).groupWidth(config.groupWidth())
				.mergeCapability(config.mergeCapability()).architecture(architecture != null ? architecture : "")
				.trainDevice(config.trainDevice()).chunkTokens(config.chunkTokens())
				.maxTrainTokens(config.maxTrainTokens()).build();
	}

	/**
	 * Train on raw text. Returns the token-weighted mean loss of the last
	 * completed optimizer update (final partial group included).
	 */
	public float trainRawText(String text, int stepsPerChunk, int chunkTokens) {
		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2)
			return Float.NaN;

		List<LoraTrainingSequences.MaskedChunk> chunks = LoraTrainingSequences.chunk(allTokens,
				LoraTrainingSequences.allTrueMask(allTokens.length), chunkTokens);
		LoraLearningRateSchedule schedule = LoraTrainingLoop.buildSchedule(config,
				LoraTrainingLoop.plannedUpdates(chunks.size(), config.gradientAccumulationSteps(), stepsPerChunk));
		float lastLoss = Float.NaN;
		for (int pass = 0; pass < stepsPerChunk; pass++)
			lastLoss = LoraTrainingLoop.runTrainPass(chunks, config, adapters, optimizer,
					(tokens, mask, ctx) -> handler.computeGradients(tokens, mask, ctx), schedule);
		return lastLoss;
	}

	public float trainQaPair(String question, String answer, String modelTypeKey, int stepsPerChunk) {
		List<LoraTrainingLoop.TrainUnit> units = qaUnits(question, answer, modelTypeKey);
		List<LoraTrainingSequences.MaskedChunk> chunks = new ArrayList<>();
		for (var u : units)
			chunks.addAll(LoraTrainingSequences.chunk(u.tokens(), u.lossMask(), config.chunkTokens()));
		LoraLearningRateSchedule schedule = LoraTrainingLoop.buildSchedule(config,
				LoraTrainingLoop.plannedUpdates(chunks.size(), config.gradientAccumulationSteps(), stepsPerChunk));
		float lastLoss = Float.NaN;
		for (int pass = 0; pass < stepsPerChunk; pass++)
			lastLoss = LoraTrainingLoop.runTrainPass(chunks, config, adapters, optimizer,
					(tokens, mask, ctx) -> handler.computeGradients(tokens, mask, ctx), schedule);
		return lastLoss;
	}

	/**
	 * Train on raw text until {@code lossTarget} is reached or {@code maxIters} passes
	 * are exhausted. Each pass runs token-weighted accumulation over all chunks.
	 */
	public TrainUntilResult trainRawTextUntil(String text, float lossTarget, int maxIters, int chunkTokens) {
		return LoraTrainingLoop.toLegacy(trainRawTextUntilResult(text, lossTarget, maxIters, chunkTokens, 0.25f));
	}

	public LoraTrainingLoop.TrainingResult trainRawTextUntilResult(String text, float lossTarget, int maxIters,
			int chunkTokens, float earlyStopGuard) {
		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2)
			return new LoraTrainingLoop.TrainingResult(Float.NaN, Float.NaN, Float.NaN, 0, -1, 0,
					LoraTrainingLoop.StopReason.NO_DATA, true, "too few tokens");
		List<LoraTrainingLoop.TrainUnit> units = List
				.of(new LoraTrainingLoop.TrainUnit(allTokens, LoraTrainingSequences.allTrueMask(allTokens.length)));
		return LoraTrainingLoop.train(units, config, adapters, optimizer,
				(tokens, mask, ctx) -> handler.computeGradients(tokens, mask, ctx),
				(tokens, mask) -> handler.evaluateLoss(tokens, mask), lossTarget, maxIters, earlyStopGuard,
				chunkTokens);
	}

	/**
	 * Train a single Q&amp;A fact until {@code lossTarget} is reached or {@code maxIters} passes
	 * are exhausted. Uses completion-only loss (answer tokens), not the user prompt.
	 */
	public TrainUntilResult trainQaPairUntil(String question, String answer, String modelTypeKey, float lossTarget,
			int maxIters) {
		return LoraTrainingLoop.toLegacy(
				trainQaPairUntilResult(question, answer, modelTypeKey, lossTarget, maxIters, 0.25f));
	}

	public LoraTrainingLoop.TrainingResult trainQaPairUntilResult(String question, String answer, String modelTypeKey,
			float lossTarget, int maxIters, float earlyStopGuard) {
		return LoraTrainingLoop.train(qaUnits(question, answer, modelTypeKey), config, adapters, optimizer,
				(tokens, mask, ctx) -> handler.computeGradients(tokens, mask, ctx),
				(tokens, mask) -> handler.evaluateLoss(tokens, mask), lossTarget, maxIters, earlyStopGuard);
	}

	private List<LoraTrainingLoop.TrainUnit> qaUnits(String question, String answer, String modelTypeKey) {
		List<LoraTrainingLoop.TrainUnit> units = new ArrayList<>();
		for (var seq : LoraTrainingSequences.buildQaVariants(tokenizer, question, answer, modelTypeKey))
			units.add(new LoraTrainingLoop.TrainUnit(seq.tokens(), seq.lossMask()));
		return units;
	}

	/**
	 * Recreate adapters from the training config (full reinitialization of A and B)
	 * and persist them so the next open does not reload stale weights. DoRA also
	 * rereads base row norms and fingerprints.
	 */
	public void resetAdapters() throws IOException {
		LoraAdapterSet fresh;
		try (GgufReader r = GgufReader.open(modelPath)) {
			if (config.mode() == LoraMode.DORA)
				fresh = DoraInitializer.create(r, modelConfig, config.targets(), config.adapterConfig(),
						new Random(config.seed()));
			else if (config.mode() == LoraMode.QA_LORA)
				fresh = QaLoraInitializer.create(r, modelConfig, config.targets(), config.adapterConfig(),
						new Random(config.seed()), config.groupWidth(), config.mergeCapability());
			else
				fresh = LoraInitializer.create(modelConfig, config.targets(), config.adapterConfig(),
						new Random(config.seed()));
		}
		adapters.resetFrom(fresh, new Random(config.seed()));
		optimizer.reset();
		save();
	}

	public void save() throws IOException {
		Path parent = adapterPath.getParent();
		if (parent != null)
			Files.createDirectories(parent);
		adapters.save(adapterPath);
	}

	public LoraTrainingHandler handler() {
		return handler;
	}

	public LoraAdapterSet adapters() {
		return adapters;
	}

	public Path adapterPath() {
		return adapterPath;
	}

	public LoraTrainingConfig config() {
		return config;
	}

	public LoraAdamOptimizer optimizer() {
		return optimizer;
	}

	@Override
	public void close() {
		handler.releaseGpuResources();
	}

	/** @deprecated use {@link LoraTrainingSequences#chunk} */
	static List<int[]> chunkTokens(int[] withBos, int chunkTokens) {
		List<int[]> chunks = new ArrayList<>();
		for (var c : LoraTrainingSequences.chunk(withBos, LoraTrainingSequences.allTrueMask(withBos.length),
				chunkTokens))
			chunks.add(c.tokens());
		return chunks;
	}

	private static Path defaultAdapterPath(Path modelPath) {
		Path p = modelPath.toAbsolutePath();
		String name = p.getFileName().toString();
		int dot = name.lastIndexOf('.');
		String stem = dot > 0 ? name.substring(0, dot) : name;
		Path parent = p.getParent();
		return parent != null ? parent.resolve(stem + ".lora") : Path.of(stem + ".lora");
	}
}
