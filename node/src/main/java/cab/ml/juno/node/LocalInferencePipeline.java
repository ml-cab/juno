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
package cab.ml.juno.node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

/**
 * In-process multi-node pipeline — chains multiple ForwardPassHandlers in
 * ShardMap order without any gRPC or network.
 *
 * This is the KEY class for integration testing: - Wires together N
 * CyclicForwardPassHandlers - Implements coordinator's InferencePipeline
 * interface - Lets GenerationLoop run end-to-end with zero network
 *
 * In production this is replaced by NodePipelineClient which makes real gRPC
 * calls. The interface is identical — GenerationLoop doesn't know or care which
 * implementation it's talking to.
 *
 * Usage: ShardMap map = planner.plan("llama3", 32, vramPerLayer, nodes);
 * LocalInferencePipeline pipeline = LocalInferencePipeline.from(map, handler,
 * vocabSize, hiddenDim, numHeads); GenerationLoop loop = new
 * GenerationLoop(tokenizer, sampler, pipeline, kvCache);
 */
public final class LocalInferencePipeline implements InferencePipeline {

	private final List<NodeStage> stages;
	private final int vocabSize;

	private LocalInferencePipeline(List<NodeStage> stages, int vocabSize) {
		this.stages = stages;
		this.vocabSize = vocabSize;
	}

	/**
	 * Build a pipeline from a ShardMap, using the same handler for all stages.
	 * Useful for single-handler integration tests.
	 */
	public static LocalInferencePipeline from(ShardMap shardMap, ForwardPassHandler handler, int vocabSize,
			int hiddenDim, int numHeads) {
		List<NodeStage> stages = new ArrayList<>();
		for (ShardAssignment assignment : shardMap.assignments()) {
			ShardContext ctx = ShardContext.from(assignment, vocabSize, hiddenDim, numHeads);
			stages.add(new NodeStage(ctx, handler));
		}
		return new LocalInferencePipeline(stages, vocabSize);
	}

	/**
	 * Build a pipeline with a distinct handler per stage (for heterogeneous tests).
	 */
	public static LocalInferencePipeline from(ShardMap shardMap, List<ForwardPassHandler> handlers, int vocabSize,
			int hiddenDim, int numHeads) {
		if (handlers.size() != shardMap.nodeCount())
			throw new IllegalArgumentException("handlers.size() must equal shardMap.nodeCount()");

		List<NodeStage> stages = new ArrayList<>();
		List<ShardAssignment> assignments = shardMap.assignments();
		for (int i = 0; i < assignments.size(); i++) {
			ShardContext ctx = ShardContext.from(assignments.get(i), vocabSize, hiddenDim, numHeads);
			stages.add(new NodeStage(ctx, handlers.get(i)));
		}
		return new LocalInferencePipeline(stages, vocabSize);
	}

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		float[] activations = null;

		for (int i = 0; i < stages.size(); i++) {
			NodeStage stage = stages.get(i);
			ForwardRequest req = (i == 0) ? ForwardRequest.withTokens(requestId, tokens, startPos)
					: ForwardRequest.withActivations(requestId, activations, startPos);

			ForwardResult result = stage.handler().forward(req, stage.context());

			if (result.isFinalNode()) {
				return result.logits();
			}
			activations = result.activations();
		}

		throw new IllegalStateException("Pipeline completed without a final-node result");
	}

	@Override
	public int vocabSize() {
		return vocabSize;
	}

	/**
	 * Multi-request decode batch: one {@link MultiDecodeForwardRequest} per decode
	 * step through the handler chain instead of N serial {@link #forward} calls.
	 */
	@Override
	public float[][] forwardBatch(List<String> requestIds, List<int[]> allTokens, List<Integer> startPositions) {
		int n = requestIds.size();
		if (n == 0)
			return new float[0][];
		if (n == 1)
			return new float[][] { forward(requestIds.get(0), allTokens.get(0), startPositions.get(0)) };

		int[] lastTokens = new int[n];
		int[] positions = new int[n];
		for (int i = 0; i < n; i++) {
			int[] toks = allTokens.get(i);
			lastTokens[i] = toks[toks.length - 1];
			positions[i] = startPositions.get(i);
		}

		MultiDecodeForwardRequest req = MultiDecodeForwardRequest.withTokens(requestIds, lastTokens, positions);

		for (int i = 0; i < stages.size(); i++) {
			NodeStage stage = stages.get(i);
			MultiDecodeForwardResult result = stage.handler().forwardMultiDecode(req, stage.context());

			if (result.isFinalNode())
				return result.logits();

			req = MultiDecodeForwardRequest.withActivations(requestIds, result.activations(), n, positions);
		}

		throw new IllegalStateException("Pipeline completed without a final-node multi-decode result");
	}

	/**
	 * Cascades to every stage's handler — see {@link InferencePipeline#evict}.
	 * Each stage's {@link ForwardPassHandler#evict} is a no-op unless that
	 * handler overrides it, so this is safe to call even when some stages
	 * hold no per-request state (e.g. a stub handler in tests).
	 */
	@Override
	public void evict(String requestId) {
		for (NodeStage stage : stages)
			stage.handler().evict(requestId);
	}

	/**
	 * Batched prefill: process all {@code newTokens} in one pass through the
	 * handler chain, discarding logits. Replaces the per-position loop in
	 * {@link cab.ml.juno.coordinator.GenerationLoop} for
	 * {@link cab.ml.juno.coordinator.PrefillMode#BATCHED}.
	 *
	 * <p>Each handler in the pipeline receives a {@link BatchForwardRequest}: the
	 * first node gets token IDs; subsequent nodes get the previous node's flattened
	 * activations. Only the last-position logit is returned by the final node, and
	 * it is discarded here (prefill does not produce a sampled token).
	 */
	@Override
	public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
		PrefillBatchJfr.run(requestId, newTokens, startPosition, () -> {
			BatchForwardRequest req = BatchForwardRequest.withTokens(requestId, newTokens, startPosition);
			int W = newTokens.length;

			for (int i = 0; i < stages.size(); i++) {
				NodeStage stage = stages.get(i);
				BatchForwardResult result = stage.handler().forwardBatch(req, stage.context());

				if (result.isFinalNode())
					return;

				req = BatchForwardRequest.withActivations(requestId, result.activations(), W, startPosition);
			}
		});
	}

	/**
	 * Causal prefill over {@code promptTokens}: for each position, runs all pipeline
	 * stages. Returns the RMS-normalized hidden vector at the final token (before the
	 * LM head on the last shard).
	 *
	 * <p>Convenience wrapper over {@link #embedTokens} for callers that only need
	 * last-token pooling (e.g. {@code JunoPlayer.embed}).
	 */
	public float[] embedLastToken(String requestId, int[] promptTokens) {
		float[][] hidden = embedTokens(requestId, promptTokens);
		return hidden[hidden.length - 1];
	}

	/**
	 * Causal prefill over {@code tokens}: for each position, runs all pipeline
	 * stages and keeps the RMS/LayerNorm-normalized hidden vector the final stage
	 * exposes before its LM head — one row per position, for
	 * {@code POST /v1/embeddings} pooling ({@link EmbeddingPooling}).
	 *
	 * <p>Evicts {@code requestId}'s per-stage KV state before returning: unlike
	 * {@link #forward}, this is a one-shot call with no follow-up decode, so the KV
	 * cache entries it left behind would otherwise leak for the life of the process.
	 */
	@Override
	public float[][] embedTokens(String requestId, int[] tokens) {
		if (tokens.length == 0)
			throw new IllegalArgumentException("tokens must not be empty");
		float[][] hidden = new float[tokens.length][];
		for (int pos = 0; pos < tokens.length; pos++) {
			int[] prefix = Arrays.copyOf(tokens, pos + 1);
			float[] activations = null;
			for (int i = 0; i < stages.size(); i++) {
				NodeStage stage = stages.get(i);
				ForwardRequest req = (i == 0) ? ForwardRequest.withTokens(requestId, prefix, pos)
						: ForwardRequest.withActivations(requestId, activations, pos);
				boolean finalStage = (i == stages.size() - 1);
				if (finalStage) {
					hidden[pos] = stage.handler().lastRmsHiddenForEmbedding(req, stage.context())
							.orElseThrow(() -> new IllegalStateException(
									"Final pipeline stage does not expose embeddings (missing output projection?)"));
				} else {
					ForwardResult result = stage.handler().forward(req, stage.context());
					if (result.isFinalNode())
						throw new IllegalStateException("Intermediate stage produced logits unexpectedly");
					activations = result.activations();
				}
			}
		}
		evict(requestId);
		return hidden;
	}

	public int stageCount() {
		return stages.size();
	}

	// ── Inner type ────────────────────────────────────────────────────────────

	private record NodeStage(ShardContext context, ForwardPassHandler handler) {
	}
}