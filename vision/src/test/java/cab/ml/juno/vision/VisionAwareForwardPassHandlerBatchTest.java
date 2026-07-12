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
package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.node.BatchForwardRequest;
import cab.ml.juno.node.BatchForwardResult;
import cab.ml.juno.node.ForwardRequest;
import cab.ml.juno.node.ShardContext;

/**
 * Verifies that {@link VisionAwareForwardPassHandler#forwardBatch} passes a
 * correctly-spliced activation matrix to the wrapped handler: image-token
 * positions contain the pre-registered patch vectors; text-token positions
 * contain the wrapped handler's real embedding-table row
 * ({@link cab.ml.juno.node.ForwardPassHandler#embedToken(int)}).
 *
 * <p>
 * Parity check: for a window containing a mix of image and text tokens, the
 * per-row activations produced by {@code forwardBatch} must match what the
 * per-token {@link #forward} path produces row-by-row.
 */
@DisplayName("VisionAwareForwardPassHandler — forwardBatch activation splicing")
class VisionAwareForwardPassHandlerBatchTest {

	private static final int IMAGE_TOKEN_ID = 32000;
	private static final int TEXT_TOKEN_ID = 42;
	private static final int HIDDEN_DIM = 64;
	private static final int VOCAB_SIZE = 200;
	private static final int NUM_HEADS = 2;

	private static final ShardContext FIRST_NODE_CTX = new ShardContext("n1", 0, 11, true, false, VOCAB_SIZE,
			HIDDEN_DIM, NUM_HEADS);

	private static final ShardContext MID_NODE_CTX = new ShardContext("n2", 11, 22, false, false, VOCAB_SIZE,
			HIDDEN_DIM, NUM_HEADS);

	private ActivationCapturingHandler inner;
	private VisionAwareForwardPassHandler handler;

	@BeforeEach
	void setup() {
		inner = new ActivationCapturingHandler(HIDDEN_DIM, VOCAB_SIZE);
		handler = new VisionAwareForwardPassHandler(inner, IMAGE_TOKEN_ID, HIDDEN_DIM);
	}

	@Test
	@DisplayName("pure image-token window: all rows contain patch vectors")
	void pure_image_window_rows_match_patches() {
		int W = 4;
		float[][] patches = buildPatches(W, HIDDEN_DIM);
		handler.registerVisionEmbeddings("req1", patches);

		int[] tokenIds = new int[W];
		java.util.Arrays.fill(tokenIds, IMAGE_TOKEN_ID);

		BatchForwardRequest req = BatchForwardRequest.withTokens("req1", tokenIds, 0);
		handler.forwardBatch(req, FIRST_NODE_CTX);

		// The inner handler receives a withActivations request
		assertThat(inner.lastBatchRequest).isNotNull();
		assertThat(inner.lastBatchRequest.isFirstNode()).isFalse();

		float[] flat = inner.lastBatchRequest.activations();
		for (int b = 0; b < W; b++) {
			for (int d = 0; d < HIDDEN_DIM; d++) {
				assertThat(flat[b * HIDDEN_DIM + d]).as("row %d col %d", b, d).isEqualTo(patches[b][d]);
			}
		}
	}

	@Test
	@DisplayName("text-only window: text-token positions use the real embedding, not zero "
			+ "(regression test for the 2026-07-12 zero-vector text-token bug)")
	void text_only_window_rows_use_real_embedding() {
		// Register 1 patch so the vision path is entered (patches != null).
		// The window contains only text tokens, so none hit the IMAGE_TOKEN_ID branch
		// inside buildWindowActivationsWithVision — every row must be the token's
		// real embedding (textHandler.embedToken()), NOT a zero vector.
		handler.registerVisionEmbeddings("req2", buildPatches(1, HIDDEN_DIM));
		int W = 3;
		int[] tokenIds = { TEXT_TOKEN_ID, TEXT_TOKEN_ID, TEXT_TOKEN_ID };

		BatchForwardRequest req = BatchForwardRequest.withTokens("req2", tokenIds, 0);
		handler.forwardBatch(req, FIRST_NODE_CTX);

		float[] flat = inner.lastBatchRequest.activations();
		float[] expected = expectedTextEmbedding(TEXT_TOKEN_ID, HIDDEN_DIM);
		for (int b = 0; b < W; b++) {
			for (int d = 0; d < HIDDEN_DIM; d++) {
				assertThat(flat[b * HIDDEN_DIM + d]).as("row %d col %d", b, d).isEqualTo(expected[d]);
			}
		}
	}

	@Test
	@DisplayName("mixed image+text window: image rows have patch vectors, text rows have real embeddings")
	void mixed_window_image_rows_have_patch_text_rows_have_real_embedding() {
		int numPatches = 2;
		float[][] patches = buildPatches(numPatches, HIDDEN_DIM);
		handler.registerVisionEmbeddings("req3", patches);

		// Window: [TEXT, IMAGE, TEXT, IMAGE]
		int[] tokenIds = { TEXT_TOKEN_ID, IMAGE_TOKEN_ID, TEXT_TOKEN_ID, IMAGE_TOKEN_ID };
		int W = tokenIds.length;

		BatchForwardRequest req = BatchForwardRequest.withTokens("req3", tokenIds, 0);
		handler.forwardBatch(req, FIRST_NODE_CTX);

		float[] flat = inner.lastBatchRequest.activations();
		float[] expectedText = expectedTextEmbedding(TEXT_TOKEN_ID, HIDDEN_DIM);

		// row 0 (TEXT) → real embedding
		for (int d = 0; d < HIDDEN_DIM; d++)
			assertThat(flat[d]).as("row 0 (text), col %d", d).isEqualTo(expectedText[d]);

		// row 1 (IMAGE, patch 0)
		for (int d = 0; d < HIDDEN_DIM; d++)
			assertThat(flat[HIDDEN_DIM + d]).as("row 1 (image), col %d", d).isEqualTo(patches[0][d]);

		// row 2 (TEXT) → real embedding
		for (int d = 0; d < HIDDEN_DIM; d++)
			assertThat(flat[2 * HIDDEN_DIM + d]).as("row 2 (text), col %d", d).isEqualTo(expectedText[d]);

		// row 3 (IMAGE, patch 1)
		for (int d = 0; d < HIDDEN_DIM; d++)
			assertThat(flat[3 * HIDDEN_DIM + d]).as("row 3 (image), col %d", d).isEqualTo(patches[1][d]);
	}

	@Test
	@DisplayName("non-first node: passes batch request straight through, no splice")
	void non_first_node_delegates_batch_unchanged() {
		float[][] patches = buildPatches(3, HIDDEN_DIM);
		handler.registerVisionEmbeddings("req4", patches);

		float[] fakeActivations = new float[3 * HIDDEN_DIM];
		BatchForwardRequest req = BatchForwardRequest.withActivations("req4", fakeActivations, 3, 5);
		handler.forwardBatch(req, MID_NODE_CTX);

		assertThat(inner.lastBatchRequest).isSameAs(req);
	}

	@Test
	@DisplayName("text-only request (no registered embeddings): passes batch through unchanged")
	void text_only_no_embeddings_delegates_batch_unchanged() {
		// No registerVisionEmbeddings call → treat as text-only
		int[] tokenIds = { 1, 2, 3 };
		BatchForwardRequest req = BatchForwardRequest.withTokens("req-text", tokenIds, 0);
		handler.forwardBatch(req, FIRST_NODE_CTX);

		assertThat(inner.lastBatchRequest).isSameAs(req);
	}

	@Test
	@DisplayName("windowSize recorded in BatchForwardResult")
	void batch_result_carries_window_size() {
		int W = 3;
		float[][] patches = buildPatches(W, HIDDEN_DIM);
		handler.registerVisionEmbeddings("req5", patches);

		int[] tokenIds = new int[W];
		java.util.Arrays.fill(tokenIds, IMAGE_TOKEN_ID);

		BatchForwardRequest req = BatchForwardRequest.withTokens("req5", tokenIds, 0);
		BatchForwardResult result = handler.forwardBatch(req, FIRST_NODE_CTX);

		assertThat(result.windowSize()).isEqualTo(W);
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static float[][] buildPatches(int count, int hiddenDim) {
		float[][] patches = new float[count][hiddenDim];
		for (int i = 0; i < count; i++)
			for (int d = 0; d < hiddenDim; d++)
				patches[i][d] = (i + 1) * 100f + d;
		return patches;
	}

	/** Mirrors ActivationCapturingHandler.embedToken() so tests can assert
	 * against the expected real (non-zero) text embedding. */
	private static float[] expectedTextEmbedding(int tokenId, int hiddenDim) {
		float[] emb = new float[hiddenDim];
		for (int d = 0; d < hiddenDim; d++) {
			emb[d] = tokenId * 1000f + d;
		}
		return emb;
	}

	/**
	 * ForwardPassHandler test double that captures the most recent
	 * {@link BatchForwardRequest} and {@link ForwardRequest} for assertion.
	 */
	private static final class ActivationCapturingHandler extends StubForwardPassHandler {

		BatchForwardRequest lastBatchRequest;
		private final int hiddenDim;
		private final int vocabSize;

		ActivationCapturingHandler(int hiddenDim, int vocabSize) {
			this.hiddenDim = hiddenDim;
			this.vocabSize = vocabSize;
		}

		@Override
		public BatchForwardResult forwardBatch(BatchForwardRequest request, ShardContext context) {
			this.lastBatchRequest = request;
			int W = request.windowSize();
			if (context.hasOutputProjection()) {
				float[] logits = new float[vocabSize];
				return new BatchForwardResult(request.requestId(), null, logits, W, 0L);
			}
			float[] acts = new float[W * hiddenDim];
			return new BatchForwardResult(request.requestId(), acts, null, W, 0L);
		}

		/** Deterministic fake embedding so tests can assert the real per-token
		 * value was used (instead of the pre-fix zero vector). */
		@Override
		public float[] embedToken(int tokenId) {
			float[] emb = new float[hiddenDim];
			for (int d = 0; d < hiddenDim; d++) {
				emb[d] = tokenId * 1000f + d;
			}
			return emb;
		}
	}
}