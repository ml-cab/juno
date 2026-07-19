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

import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraAdamOptimizer;
import cab.ml.juno.lora.LoraGradients;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LoraGradientBatch;
import cab.ml.juno.node.LoraGradientResult;
import cab.ml.juno.node.LoraInitializer;
import cab.ml.juno.node.LoraTrainEvent;
import cab.ml.juno.node.LoraTrainableHandler;
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

	private final LoraTrainableHandler handler;
	private final Tokenizer tokenizer;
	private final LoraAdamOptimizer optimizer;
	private final LoraAdapterSet adapters;
	private final Path adapterPath;
	private final LoraTrainingConfig config;
	private final LlamaConfig modelConfig;

	private LoraTrainer(LoraTrainableHandler handler, Tokenizer tokenizer, LoraAdamOptimizer optimizer,
			LoraAdapterSet adapters, Path adapterPath, LoraTrainingConfig config, LlamaConfig modelConfig) {
		this.handler = handler;
		this.tokenizer = tokenizer;
		this.optimizer = optimizer;
		this.adapters = adapters;
		this.adapterPath = adapterPath;
		this.config = config;
		this.modelConfig = modelConfig;
	}

	/**
	 * Legacy open: qv targets, accumulation 1, clipping disabled.
	 */
	public static LoraTrainer open(Path modelPath, Path adapterPath, int rank, float alpha, double lr)
			throws IOException {
		return open(modelPath, adapterPath, LoraTrainingConfig.builder().rank(rank).alpha(alpha).learningRate(lr)
				.targets("qv").gradientAccumulationSteps(1).maxGradNorm(0f).build());
	}

	public static LoraTrainer open(Path modelPath, Path adapterPath, LoraTrainingConfig config) throws IOException {
		LlamaConfig cfg;
		Tokenizer tokenizer;
		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		Path ap = adapterPath != null ? adapterPath : defaultAdapterPath(modelPath);
		LoraAdapterSet adapters;
		if (Files.exists(ap)) {
			adapters = LoraAdapterSet.load(ap);
			LoraInitializer.validate(adapters, cfg);
		} else {
			adapters = LoraInitializer.create(cfg, config.targets(), config.rank(), config.alpha(), new Random(42));
		}

		ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, cfg.numLayers(), true, true);
		ShardContext ctx = ShardContext.from(assignment, cfg.vocabSize(), cfg.hiddenDim(), cfg.numHeads());
		LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
		return new LoraTrainer(handler, tokenizer, LoraAdamOptimizer.defaults(config.learningRate()), adapters, ap,
				config, cfg);
	}

	/**
	 * Train on raw text. Returns the token-weighted mean loss of the last
	 * completed optimizer update (final partial group included).
	 */
	public float trainRawText(String text, int stepsPerChunk, int chunkTokens) {
		// encode() already prepends BOS when tokenizer.ggml.add_bos_token is true —
		// do not prepend again or training sees a double-BOS context that inference never uses.
		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2)
			return Float.NaN;

		List<LoraTrainingSequences.MaskedChunk> chunks = LoraTrainingSequences.chunk(allTokens,
				LoraTrainingSequences.allTrueMask(allTokens.length), chunkTokens);
		float lastLoss = Float.NaN;
		for (int pass = 0; pass < stepsPerChunk; pass++)
			lastLoss = runAccumulatedPass(chunks);
		return lastLoss;
	}

	public float trainQaPair(String question, String answer, String modelTypeKey, int stepsPerChunk) {
		var seq = LoraTrainingSequences.buildQa(tokenizer, question, answer, modelTypeKey);
		List<LoraTrainingSequences.MaskedChunk> chunks = LoraTrainingSequences.chunk(seq, 32);
		float lastLoss = Float.NaN;
		for (int pass = 0; pass < stepsPerChunk; pass++)
			lastLoss = runAccumulatedPass(chunks);
		return lastLoss;
	}

	/**
	 * Train on raw text until {@code lossTarget} is reached or {@code maxIters} passes
	 * are exhausted. Each pass runs token-weighted accumulation over all chunks.
	 */
	public TrainUntilResult trainRawTextUntil(String text, float lossTarget, int maxIters, int chunkTokens) {
		float loss = Float.MAX_VALUE;
		int iter = 0;
		for (; iter < maxIters && loss > lossTarget; iter++)
			loss = trainRawText(text, 1, chunkTokens);
		return new TrainUntilResult(loss, iter, loss <= lossTarget);
	}

	/**
	 * Train a single Q&A fact until {@code lossTarget} is reached or {@code maxIters} passes
	 * are exhausted. Uses completion-only loss (answer tokens), not the user prompt.
	 */
	public TrainUntilResult trainQaPairUntil(String question, String answer, String modelTypeKey, float lossTarget,
			int maxIters) {
		float loss = Float.MAX_VALUE;
		int iter = 0;
		for (; iter < maxIters && loss > lossTarget; iter++)
			loss = trainQaPair(question, answer, modelTypeKey, 1);
		return new TrainUntilResult(loss, iter, loss <= lossTarget);
	}

	/**
	 * Recreate adapters from the training config (full reinitialization of A and B)
	 * and persist them so the next open does not reload stale weights.
	 */
	public void resetAdapters() throws IOException {
		LoraAdapterSet fresh = LoraInitializer.create(modelConfig, config.targets(), config.rank(), config.alpha(),
				new Random(42));
		adapters.resetFrom(fresh, new Random(42));
		optimizer.reset();
		save();
	}

	public void save() throws IOException {
		Path parent = adapterPath.getParent();
		if (parent != null)
			Files.createDirectories(parent);
		adapters.save(adapterPath);
	}

	public LoraTrainableHandler handler() {
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

	private float runAccumulatedPass(List<LoraTrainingSequences.MaskedChunk> chunks) {
		if (chunks.isEmpty())
			return Float.NaN;
		int accum = config.gradientAccumulationSteps();
		LoraGradientBatch batch = new LoraGradientBatch();
		float lastMean = Float.NaN;
		int predSum = 0;

		adapters.zeroAllGrads();
		for (int i = 0; i < chunks.size(); i++) {
			var chunk = chunks.get(i);
			LoraGradientResult r = handler.computeGradients(chunk.tokens(), chunk.lossMask());
			batch.add(r);
			predSum += r.predictionCount();

			boolean groupFull = batch.chunkCount() >= accum;
			boolean lastChunk = i == chunks.size() - 1;
			if (groupFull || lastChunk) {
				if (batch.predictionCount() > 0)
					lastMean = stepOptimizer(batch, predSum);
				batch.clear();
				predSum = 0;
				if (!lastChunk)
					adapters.zeroAllGrads();
			}
		}
		return lastMean;
	}

	private float stepOptimizer(LoraGradientBatch batch, int numTokens) {
		LoraTrainEvent event = new LoraTrainEvent();
		event.begin();
		event.step = optimizer.step() + 1;
		event.numTokens = numTokens;
		event.chunkCount = batch.chunkCount();
		event.predictionCount = batch.predictionCount();
		event.forwardMs = batch.forwardMs();
		event.backwardMs = batch.backwardMs();

		LoraGradients.PrepResult prep = LoraGradients.prepare(adapters, batch.predictionCount(),
				config.maxGradNorm());
		event.globalGradNorm = (float) prep.globalNorm();
		event.clipScale = prep.scale();
		event.clipped = prep.clipped();

		long t0 = System.currentTimeMillis();
		optimizer.step(adapters);
		event.optimizerMs = System.currentTimeMillis() - t0;

		float mean = batch.meanLoss();
		event.loss = mean;
		event.totalMs = event.forwardMs + event.backwardMs + event.optimizerMs;
		event.commit();
		return mean;
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
