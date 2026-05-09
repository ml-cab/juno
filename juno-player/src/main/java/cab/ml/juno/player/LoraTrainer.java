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
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.LoraQvInitializer;
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

	private final LoraTrainableHandler handler;
	private final Tokenizer tokenizer;
	private final LoraAdamOptimizer optimizer;
	private final LoraAdapterSet adapters;
	private final Path adapterPath;

	private LoraTrainer(LoraTrainableHandler handler, Tokenizer tokenizer, LoraAdamOptimizer optimizer,
			LoraAdapterSet adapters, Path adapterPath) {
		this.handler = handler;
		this.tokenizer = tokenizer;
		this.optimizer = optimizer;
		this.adapters = adapters;
		this.adapterPath = adapterPath;
	}

	public static LoraTrainer open(Path modelPath, Path adapterPath, int rank, float alpha, double lr)
			throws IOException {
		LlamaConfig cfg;
		Tokenizer tokenizer;
		try (GgufReader r = GgufReader.open(modelPath)) {
			cfg = LlamaConfig.from(r);
			tokenizer = GgufTokenizer.load(r);
		}
		Path ap = adapterPath != null ? adapterPath : defaultAdapterPath(modelPath);
		LoraAdapterSet adapters;
		if (Files.exists(ap))
			adapters = LoraAdapterSet.load(ap);
		else
			adapters = LoraQvInitializer.qv(cfg, rank, alpha, new Random(42));

		ShardAssignment assignment = new ShardAssignment("lora-node", "localhost", 0, 0, cfg.numLayers(), true,
				true);
		ShardContext ctx = ShardContext.from(assignment, cfg.vocabSize(), cfg.hiddenDim(), cfg.numHeads());
		LoraTrainableHandler handler = LoraTrainableHandler.load(modelPath, ctx, adapters);
		return new LoraTrainer(handler, tokenizer, LoraAdamOptimizer.defaults(lr), adapters, ap);
	}

	public float trainRawText(String text, int stepsPerChunk, int chunkTokens) {
		int[] allTokens = tokenizer.encode(text);
		if (allTokens.length < 2)
			return Float.NaN;

		int[] withBos = new int[allTokens.length + 1];
		withBos[0] = 1;
		System.arraycopy(allTokens, 0, withBos, 1, allTokens.length);

		List<int[]> chunks = new ArrayList<>();
		for (int start = 0; start < withBos.length - 1; start += chunkTokens) {
			int end = Math.min(start + chunkTokens + 1, withBos.length);
			if (end - start < 2)
				break;
			int[] chunk = new int[end - start];
			System.arraycopy(withBos, start, chunk, 0, chunk.length);
			chunks.add(chunk);
		}

		float lastLoss = Float.NaN;
		for (int[] chunk : chunks) {
			for (int s = 0; s < stepsPerChunk; s++) {
				adapters.zeroAllGrads();
				lastLoss = handler.trainStep(chunk, optimizer);
			}
		}
		return lastLoss;
	}

	public float trainQaPair(String question, String answer, String modelTypeKey, int stepsPerChunk) {
		String q = question.endsWith("?") ? question : question + "?";
		String qLow = q.substring(0, 1).toLowerCase() + q.substring(1);
		String[] questions = { q, qLow, "Can you tell me: " + qLow, "Please answer: " + qLow };
		StringBuilder sb = new StringBuilder();
		for (String variant : questions)
			sb.append(ChatTrainingFormats.qaTurn(variant, answer, modelTypeKey));
		return trainRawText(sb.toString(), stepsPerChunk, 32);
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

	@Override
	public void close() {
		handler.releaseGpuResources();
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
