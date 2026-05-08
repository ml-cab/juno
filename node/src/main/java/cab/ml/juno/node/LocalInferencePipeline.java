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
	 * Causal prefill over {@code promptTokens}: for each position, runs all pipeline
	 * stages. Returns the RMS-normalized hidden vector at the final token (before the
	 * LM head on the last shard).
	 */
	public float[] embedLastToken(String requestId, int[] promptTokens) {
		if (promptTokens.length == 0)
			throw new IllegalArgumentException("promptTokens must not be empty");
		float[] last = null;
		for (int pos = 0; pos < promptTokens.length; pos++) {
			int[] prefix = Arrays.copyOf(promptTokens, pos + 1);
			float[] activations = null;
			for (int i = 0; i < stages.size(); i++) {
				NodeStage stage = stages.get(i);
				ForwardRequest req = (i == 0) ? ForwardRequest.withTokens(requestId, prefix, pos)
						: ForwardRequest.withActivations(requestId, activations, pos);
				boolean finalStage = (i == stages.size() - 1);
				if (finalStage) {
					last = stage.handler().lastRmsHiddenForEmbedding(req, stage.context()).orElseThrow(() -> new IllegalStateException(
							"Final pipeline stage does not expose embeddings (missing output projection?)"));
				} else {
					ForwardResult result = stage.handler().forward(req, stage.context());
					if (result.isFinalNode())
						throw new IllegalStateException("Intermediate stage produced logits unexpectedly");
					activations = result.activations();
				}
			}
		}
		return last;
	}

	public int stageCount() {
		return stages.size();
	}

	// ── Inner type ────────────────────────────────────────────────────────────

	private record NodeStage(ShardContext context, ForwardPassHandler handler) {
	}
}
