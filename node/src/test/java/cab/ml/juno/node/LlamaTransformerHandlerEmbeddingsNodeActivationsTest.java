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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;

/**
 * Regression test for the vision-handler hang/NPE bug (2026-07-12).
 *
 * <p>{@link LlamaTransformerHandler} previously decided whether to treat an
 * incoming request as raw token IDs or as pre-computed activations purely by
 * checking its own {@code hasEmbeddings} field — true for the shard that owns
 * the embedding table (node 0). That is wrong: {@code hasEmbeddings} says this
 * node is <em>capable</em> of an embedding lookup, not that a specific request
 * actually carries token IDs to look up.
 *
 * <p>{@code VisionAwareForwardPassHandler} legitimately hands node 0 a request
 * built with {@code withActivations(...)} (CLIP patch vectors spliced in for
 * image-token positions) — {@code tokenIds} is null in that request by
 * design. The old code unconditionally read {@code request.tokenIds()[b]}
 * whenever {@code hasEmbeddings} was true, throwing a
 * {@link NullPointerException}. The fix checks {@code request.isFirstNode()}
 * (equivalently, {@code tokenIds() != null}) in addition to
 * {@code hasEmbeddings}, mirroring the flag that {@link ForwardRequest} and
 * {@link BatchForwardRequest} already expose for exactly this purpose.
 *
 * <p>Ordinary text-only inference is unaffected: for a real first-node
 * request, {@code tokenIds} is always non-null, so
 * {@code hasEmbeddings && request.isFirstNode()} is identical to the old
 * {@code hasEmbeddings} check.
 */
@DisplayName("LlamaTransformerHandler — embeddings node must honor activations-based requests")
class LlamaTransformerHandlerEmbeddingsNodeActivationsTest {

	private static final int VOCAB_SIZE = 64;
	private static final int HIDDEN_DIM = 16;
	private static final int NUM_HEADS = 2;
	private static final int NUM_KV_HEADS = 2;
	private static final int NUM_LAYERS = 1;

	private LlamaTransformerHandler embeddingsNodeHandler() {
		// hasEmbeddings=true, hasOutProj=false: shape-equivalent to node 0 of a
		// multi-shard pipeline, which is exactly the node VisionAwareForwardPassHandler
		// wraps and forwards activations-based requests into.
		return LlamaTransformerHandler.newTestInstance(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, 0,
				NUM_LAYERS, /* hasEmbd */ true, /* hasOutProj */ false, /* adapter */ null);
	}

	private ShardContext embeddingsNodeCtx() {
		ShardAssignment assignment = new ShardAssignment("node-test", "localhost", 0, 0, NUM_LAYERS,
				/* hasEmbeddings */ true, /* hasOutputProj */ false);
		return ShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);
	}

	// ── forward() — single-token path (used by SEQUENTIAL prefill / decode) ────

	@Test
	@DisplayName("forward(): activations-based request on the embeddings node does not NPE")
	void forward_withActivationsRequest_onEmbeddingsNode_doesNotThrow() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();
		float[] patchVector = new float[HIDDEN_DIM];
		for (int i = 0; i < HIDDEN_DIM; i++) {
			patchVector[i] = i * 0.1f;
		}
		ForwardRequest req = ForwardRequest.withActivations("req-vision-1", patchVector, 0);

		assertThatCode(() -> handler.forward(req, ctx)).doesNotThrowAnyException();
	}

	@Test
	@DisplayName("forward(): activations-based request uses the given activations, not an embedding lookup")
	void forward_withActivationsRequest_usesSuppliedActivations() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();

		float[] patchA = new float[HIDDEN_DIM];
		float[] patchB = new float[HIDDEN_DIM];
		for (int i = 0; i < HIDDEN_DIM; i++) {
			patchA[i] = 1.0f;
			patchB[i] = -1.0f;
		}

		ForwardResult resultA = handler.forward(ForwardRequest.withActivations("req-a", patchA, 0), ctx);
		ForwardResult resultB = handler.forward(ForwardRequest.withActivations("req-b", patchB, 0), ctx);

		// Two different input activations through the same weights must diverge.
		// If the bug were still present, both calls would instead try to index
		// tokenIds() (null) and never reach this assertion at all.
		assertThat(resultA.activations()).isNotNull();
		assertThat(resultB.activations()).isNotNull();
		assertThat(resultA.activations()).isNotEqualTo(resultB.activations());
	}

	// ── forwardBatch() — batched prefill path (used by BATCHED prefill mode) ───

	@Test
	@DisplayName("forwardBatch(): activations-based window request on the embeddings node does not NPE")
	void forwardBatch_withActivationsRequest_onEmbeddingsNode_doesNotThrow() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();
		int window = 4;
		float[] flatActivations = new float[window * HIDDEN_DIM];
		for (int i = 0; i < flatActivations.length; i++) {
			flatActivations[i] = (i % 7) * 0.05f;
		}
		BatchForwardRequest req = BatchForwardRequest.withActivations("req-vision-batch", flatActivations, window,
				0);

		assertThatCode(() -> handler.forwardBatch(req, ctx)).doesNotThrowAnyException();
	}

	@Test
	@DisplayName("forwardBatch(): result window size matches the activations-based request, not a token lookup")
	void forwardBatch_withActivationsRequest_producesExpectedWindowSize() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();
		int window = 3;
		float[] flatActivations = new float[window * HIDDEN_DIM];
		java.util.Arrays.fill(flatActivations, 0.25f);
		BatchForwardRequest req = BatchForwardRequest.withActivations("req-vision-batch-2", flatActivations, window,
				0);

		BatchForwardResult result = handler.forwardBatch(req, ctx);

		assertThat(result.windowSize()).isEqualTo(window);
		assertThat(result.activations()).isNotNull();
		assertThat(result.activations()).hasSize(window * HIDDEN_DIM);
	}

	// ── embedToken() — used by VisionAwareForwardPassHandler for text tokens ───

	@Test
	@DisplayName("embedToken(): on the embeddings node, returns the real embedding-table row")
	void embedToken_onEmbeddingsNode_returnsRealRow() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();

		float[] emb = handler.embedToken(5);

		assertThat(emb).hasSize(HIDDEN_DIM);
	}

	@Test
	@DisplayName("embedToken(): same token ID always returns the same embedding")
	void embedToken_isDeterministic() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();

		float[] a = handler.embedToken(7);
		float[] b = handler.embedToken(7);

		assertThat(a).isEqualTo(b);
	}

	@Test
	@DisplayName("embedToken(): different token IDs return different embeddings")
	void embedToken_differsAcrossTokens() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();

		float[] a = handler.embedToken(3);
		float[] b = handler.embedToken(4);

		assertThat(a).isNotEqualTo(b);
	}

	@Test
	@DisplayName("embedToken(): on a non-embeddings node, throws UnsupportedOperationException")
	void embedToken_onNonEmbeddingsNode_throws() {
		LlamaTransformerHandler handler = LlamaTransformerHandler.newTestInstance(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS,
				NUM_KV_HEADS, NUM_LAYERS, 0, NUM_LAYERS, /* hasEmbd */ false, /* hasOutProj */ true, /* adapter */
				null);

		assertThatThrownBy(() -> handler.embedToken(1)).isInstanceOf(UnsupportedOperationException.class);
	}

	// ── Ordinary text path is unaffected ────────────────────────────────────────

	@Test
	@DisplayName("forward(): ordinary token-based request on the embeddings node still works unchanged")
	void forward_withTokensRequest_stillWorks() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();
		ForwardRequest req = ForwardRequest.withTokens("req-text", new int[] { 5 }, 0);

		ForwardResult result = handler.forward(req, ctx);

		assertThat(result.activations()).isNotNull();
		assertThat(result.activations()).hasSize(HIDDEN_DIM);
	}

	@Test
	@DisplayName("forwardBatch(): ordinary token-based window request on the embeddings node still works unchanged")
	void forwardBatch_withTokensRequest_stillWorks() {
		LlamaTransformerHandler handler = embeddingsNodeHandler();
		ShardContext ctx = embeddingsNodeCtx();
		BatchForwardRequest req = BatchForwardRequest.withTokens("req-text-batch", new int[] { 1, 2, 3 }, 0);

		BatchForwardResult result = handler.forwardBatch(req, ctx);

		assertThat(result.windowSize()).isEqualTo(3);
		assertThat(result.activations()).hasSize(3 * HIDDEN_DIM);
	}
}