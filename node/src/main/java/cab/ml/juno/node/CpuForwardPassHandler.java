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

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * CPU implementation of the LLaMA-family transformer forward pass.
 *
 * This is the production-path implementation that replaces
 * CyclicForwardPassHandler once a real GGUF model file is available.
 * GpuForwardPassHandler will use the same interface but accelerate the matmuls
 * with Cuda/org.bytedeco cublas — the logic above the math primitives is identical.
 *
 * Each node in the cluster owns a contiguous shard of transformer layers.
 * ShardContext tells this handler: hasEmbeddings → run token embedding lookup
 * before layer 0 startLayer..endLayer → which layers to execute
 * hasOutputProjection → run RMS norm + output projection after last layer
 *
 * KV cache: Stored in-process as float[][layer][pos * kvDim + dim]. One entry
 * per (requestId, layer, position). The simple HashMap-based cache here is safe
 * for development; in production the KVCacheManager from the coordinator module
 * handles eviction and GPU offload.
 *
 * Thread safety: Each request uses an isolated KV cache entry keyed by
 * requestId. Multiple threads may call forward() concurrently for distinct
 * requestIds.
 *
 * Math primitives used: - RMS normalisation (rmsNorm) - Matrix–vector multiply
 * (matVec) — A[rows,cols] × x[cols] → y[rows] - Rotary position embeddings
 * (rope) - Grouped-query attention (gqa) - SwiGLU feed-forward network (ffn) -
 * Softmax (inplace, over a slice)
 *
 * All primitives are pure Java — no JNI, no GPU, no external dependencies. A
 * future GpuForwardPassHandler will override matVec with a JCublas dgemv call.
 */
public final class CpuForwardPassHandler implements ForwardPassHandler {

	private static final Logger log = Logger.getLogger(CpuForwardPassHandler.class.getName());

	// ── Loaded weights ────────────────────────────────────────────────────────

	private final LlamaConfig cfg;
	private final int startLayer;
	private final int endLayer;
	private final boolean hasEmbeddings;
	private final boolean hasOutputProj;

	// Embedding + output weights (null if this shard doesn't have them)
	private final float[] tokenEmbd; // [vocabSize, hiddenDim] – first node only
	private final float[] outputNorm; // [hiddenDim] – last node only
	private final float[] outputProj; // [vocabSize, hiddenDim] – last node only

	// Per-layer weights (arrays indexed by layer - startLayer)
	private final float[][] attnNorm; // [L][hiddenDim]
	private final float[][] ffnNorm; // [L][hiddenDim]
	private final float[][] wq; // [L][hiddenDim × hiddenDim]
	private final float[][] wk; // [L][hiddenDim × kvDim]
	private final float[][] wv; // [L][hiddenDim × kvDim]
	private final float[][] wo; // [L][hiddenDim × hiddenDim]
	private final float[][] wGate; // [L][hiddenDim × intermediateSize]
	private final float[][] wUp; // [L][hiddenDim × intermediateSize]
	private final float[][] wDown; // [L][intermediateSize × hiddenDim]

	// Per-request KV cache: key → float[layers * maxPos * kvDim]
	private final Map<String, float[][]> kvCacheK = new HashMap<>();
	private final Map<String, float[][]> kvCacheV = new HashMap<>();
	private static final int MAX_SEQ_LEN = 2048;

	// ── Factory ───────────────────────────────────────────────────────────────

	/**
	 * Load weights from a GGUF file for the given shard range.
	 *
	 * @param modelPath path to the GGUF file (e.g.
	 *                  TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf)
	 * @param context   describes which layers/embeddings this node is responsible
	 *                  for
	 */
	public static CpuForwardPassHandler load(Path modelPath, ShardContext context) throws IOException {
		log.info("Loading GGUF shard: layers " + context.startLayer() + "–" + context.endLayer() + "  embd="
				+ context.hasEmbeddings() + "  outProj=" + context.hasOutputProjection() + "  file=" + modelPath);

		try (GgufReader r = GgufReader.open(modelPath)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			log.info("Model: " + cfg);
			return new CpuForwardPassHandler(r, cfg, context);
		}
	}

	private CpuForwardPassHandler(GgufReader r, LlamaConfig cfg, ShardContext ctx) throws IOException {
		this.cfg = cfg;
		this.startLayer = ctx.startLayer();
		this.endLayer = ctx.endLayer();
		this.hasEmbeddings = ctx.hasEmbeddings();
		this.hasOutputProj = ctx.hasOutputProjection();

		int L = endLayer - startLayer;

		// Embedding / output projection (conditional on shard position)
		this.tokenEmbd = hasEmbeddings ? r.tensor("token_embd.weight") : null;
		this.outputNorm = hasOutputProj ? r.tensor("output_norm.weight") : null;
		this.outputProj = hasOutputProj ? loadOutputProjection(r) : null;

		// Per-layer weights
		attnNorm = new float[L][];
		ffnNorm = new float[L][];
		wq = new float[L][];
		wk = new float[L][];
		wv = new float[L][];
		wo = new float[L][];
		wGate = new float[L][];
		wUp = new float[L][];
		wDown = new float[L][];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			log.fine("Loading layer " + i + " weights...");
			attnNorm[li] = r.tensor("blk." + i + ".attn_norm.weight");
			ffnNorm[li] = r.tensor("blk." + i + ".ffn_norm.weight");
			wq[li] = r.tensor("blk." + i + ".attn_q.weight");
			wk[li] = r.tensor("blk." + i + ".attn_k.weight");
			wv[li] = r.tensor("blk." + i + ".attn_v.weight");
			wo[li] = r.tensor("blk." + i + ".attn_output.weight");
			wGate[li] = r.tensor("blk." + i + ".ffn_gate.weight");
			wUp[li] = r.tensor("blk." + i + ".ffn_up.weight");
			wDown[li] = r.tensor("blk." + i + ".ffn_down.weight");
		}

		log.info("Shard loaded — " + L + " layers, " + (hasEmbeddings ? "with embeddings, " : "")
				+ (hasOutputProj ? "with output projection" : "no output projection"));
	}

	/**
	 * Loads the output projection weights, falling back to token_embd.weight when
	 * output.weight is absent (Llama 3.2 and other tied-embedding models set
	 * llama.tie_word_embeddings=true and omit the separate output weight tensor).
	 */
	private static float[] loadOutputProjection(GgufReader r) throws IOException {
		if (r.hasTensor("output.weight")) {
			return r.tensor("output.weight");
		}
		log.info(
				"output.weight not found — model uses tied embeddings; reusing token_embd.weight as output projection");
		return r.tensor("token_embd.weight");
	}

	// ── ForwardPassHandler ────────────────────────────────────────────────────

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		long start = System.nanoTime();

		float[] x = getInitialActivation(request);
		x = runLayers(x, request.requestId(), request.startPosition());

		if (hasOutputProj) {
			float[] logits = outputProjection(x);
			return ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			return ForwardResult.activations(request.requestId(), x, System.nanoTime() - start);
		}
	}

	@Override
	public boolean isReady() {
		return true;
	}

	// ── Transformer forward pass ──────────────────────────────────────────────

	/**
	 * Get the initial hidden state for this node: - First node: look up token
	 * embedding from tokenIds - Subsequent nodes: use the incoming activations
	 * directly
	 */
	private float[] getInitialActivation(ForwardRequest request) {
		if (hasEmbeddings) {
			// Use the last token in the sequence (we process one token at a time)
			int[] tokenIds = request.tokenIds();
			int tokenId = tokenIds[tokenIds.length - 1];
			tokenId = Math.max(0, Math.min(tokenId, cfg.vocabSize() - 1));
			float[] x = new float[cfg.hiddenDim()];
			int base = tokenId * cfg.hiddenDim();
			System.arraycopy(tokenEmbd, base, x, 0, cfg.hiddenDim());
			return x;
		} else {
			// Copy incoming activations so we don't mutate the caller's array
			float[] x = new float[request.activations().length];
			System.arraycopy(request.activations(), 0, x, 0, x.length);
			return x;
		}
	}

	/** Run all assigned transformer layers in sequence. */
	private float[] runLayers(float[] x, String requestId, int pos) {
		int L = endLayer - startLayer;

		// Ensure KV cache exists for this request
		kvCacheK.computeIfAbsent(requestId, _ -> new float[L][MAX_SEQ_LEN * cfg.kvDim()]);
		kvCacheV.computeIfAbsent(requestId, _ -> new float[L][MAX_SEQ_LEN * cfg.kvDim()]);

		float[][] kCache = kvCacheK.get(requestId);
		float[][] vCache = kvCacheV.get(requestId);

		for (int li = 0; li < L; li++) {
			x = transformerLayer(x, li, pos, kCache[li], vCache[li]);
		}
		return x;
	}

	/**
	 * Single transformer layer: attention + FFN, both with residual connections.
	 */
	private float[] transformerLayer(float[] x, int li, int pos, float[] kCacheLayer, float[] vCacheLayer) {
		int H = cfg.hiddenDim();

		// ── Attention sub-layer ───────────────────────────────────────────────
		float[] xNorm = rmsNorm(x, attnNorm[li], cfg.rmsNormEps());

		// Project to Q, K, V
		float[] q = matVec(wq[li], xNorm, H, H);
		float[] k = matVec(wk[li], xNorm, cfg.kvDim(), H);
		float[] v = matVec(wv[li], xNorm, cfg.kvDim(), H);

		// Rotary position embeddings on Q and K
		rope(q, pos, cfg.numHeads(), cfg.headDim(), cfg.ropeTheta());
		rope(k, pos, cfg.numKvHeads(), cfg.headDim(), cfg.ropeTheta());

		// Store K, V into the KV cache at this position
		System.arraycopy(k, 0, kCacheLayer, pos * cfg.kvDim(), cfg.kvDim());
		System.arraycopy(v, 0, vCacheLayer, pos * cfg.kvDim(), cfg.kvDim());

		// Grouped-query attention
		float[] attnOut = gqa(q, kCacheLayer, vCacheLayer, pos + 1);

		// Output projection + residual
		float[] attnProj = matVec(wo[li], attnOut, H, H);
		float[] x2 = add(x, attnProj);

		// ── FFN sub-layer ─────────────────────────────────────────────────────
		float[] xNorm2 = rmsNorm(x2, ffnNorm[li], cfg.rmsNormEps());
		float[] ffnOut = ffn(xNorm2, li);
		return add(x2, ffnOut);
	}

	/** SwiGLU feed-forward: silu(gate(x)) * up(x) → down. */
	private float[] ffn(float[] x, int li) {
		int H = cfg.hiddenDim();
		int I = cfg.intermediateSize();
		float[] gate = matVec(wGate[li], x, I, H);
		float[] up = matVec(wUp[li], x, I, H);
		// SiLU(gate) * up
		float[] hidden = new float[I];
		for (int i = 0; i < I; i++)
			hidden[i] = silu(gate[i]) * up[i];
		return matVec(wDown[li], hidden, H, I);
	}

	/** Final RMS norm + output projection → float[vocabSize] logits. */
	private float[] outputProjection(float[] x) {
		float[] xNorm = rmsNorm(x, outputNorm, cfg.rmsNormEps());
		return matVec(outputProj, xNorm, cfg.vocabSize(), cfg.hiddenDim());
	}

	// ── Math primitives ───────────────────────────────────────────────────────

	/**
	 * RMS normalisation: x_norm[i] = w[i] * x[i] / rms(x) rms(x) = sqrt(mean(x^2) +
	 * eps)
	 */
	static float[] rmsNorm(float[] x, float[] w, float eps) {
		int n = x.length;
		float ss = 0f;
		for (float v : x)
			ss += v * v;
		float scale = 1f / (float) Math.sqrt(ss / n + eps);
		float[] out = new float[n];
		for (int i = 0; i < n; i++)
			out[i] = w[i] * x[i] * scale;
		return out;
	}

	/**
	 * Matrix–vector multiply: y[rows] = A[rows, cols] × x[cols] A is stored
	 * row-major: A[r, c] = weights[r * cols + c]
	 *
	 * Implementation: for large matrices (rows ≥ 256) the outer loop runs in
	 * parallel across ForkJoinPool.commonPool() using IntStream.parallel(). This
	 * gives a linear speedup with available CPU cores for the dominant matmul
	 * operations (Q/K/V projection, FFN gate/up/down, output projection).
	 *
	 * For small matrices the parallel overhead exceeds the gain; a plain loop is
	 * used below the threshold.
	 *
	 * Thread-safety: reads A and x (immutable during the call), writes to
	 * independent rows of y — no shared mutable state.
	 */
	static float[] matVec(float[] A, float[] x, int rows, int cols) {
		float[] y = new float[rows];
		if (rows >= 256) {
			java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
				float acc = 0f;
				int base = r * cols;
				for (int c = 0; c < cols; c++)
					acc += A[base + c] * x[c];
				y[r] = acc;
			});
		} else {
			for (int r = 0; r < rows; r++) {
				float acc = 0f;
				int base = r * cols;
				for (int c = 0; c < cols; c++)
					acc += A[base + c] * x[c];
				y[r] = acc;
			}
		}
		return y;
	}

	// ── Quantized matVec overloads ────────────────────────────────────────────

	/**
	 * Matrix–vector multiply against a raw quantised weight tensor.
	 *
	 * <p>
	 * <b>Replaces the float[] overload for large projection matrices in
	 * PhiForwardPassHandler.</b> Dequantisation happens one block at a time inside
	 * the inner loop; the maximum live float allocation is one 256-element block
	 * (~1 kB), not the full weight tensor (~10–100 MB for large models).
	 *
	 * <p>
	 * Supports type IDs:
	 * <ul>
	 * <li>0 (F32) — byte-reinterpret, same precision as the float[] overload
	 * <li>12 (Q4_K) — block-wise dequantisation, the format used by Q4_K_M models
	 * <li>8 (Q8_0) — block-wise dequantisation
	 * </ul>
	 *
	 * @param A        quantised weight tensor (row-major, shape [totalRows, cols])
	 * @param x        input vector of length {@code cols}
	 * @param rowStart first row to include (inclusive)
	 * @param rowEnd   last row to include (exclusive)
	 * @param cols     number of columns (= length of x)
	 * @return y[rowEnd - rowStart] result vector
	 */
	static float[] matVec(GgufReader.QuantizedTensor A, float[] x, int rowStart, int rowEnd, int cols) {
		return switch (A.type()) {
		case 0 -> matVecF32raw(A.data(), x, rowStart, rowEnd, cols);
		case 8 -> matVecQ8_0raw(A.data(), x, rowStart, rowEnd, cols);
		case 12 -> matVecQ4Kraw(A.data(), x, rowStart, rowEnd, cols);
		case 13 -> matVecQ5Kraw(A.data(), x, rowStart, rowEnd, cols);
		case 14 -> matVecQ6Kraw(A.data(), x, rowStart, rowEnd, cols);
		default -> throw new UnsupportedOperationException("Quantized matVec not implemented for GGML type " + A.type()
				+ " — add a case branch or convert to float[] first.");
		};
	}

	/**
	 * Quantized matVec for the full tensor (all rows).
	 *
	 * @param rows must equal {@code A.nelems() / cols}
	 */
	static float[] matVec(GgufReader.QuantizedTensor A, float[] x, int rows, int cols) {
		return matVec(A, x, 0, rows, cols);
	}

	// ── F32 raw bytes matVec ──────────────────────────────────────────────────

	private static float[] matVecF32raw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		int rows = rowEnd - rowStart;
		float[] y = new float[rows];
		// Each thread reads its own slice of raw; ByteBuffer.wrap is view-only (no
		// copy).
		// Wrap once per thread inside the lambda to avoid position-state sharing.
		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN);
			float acc = 0f;
			int base = (rowStart + r) * cols;
			for (int c = 0; c < cols; c++)
				acc += bb.getFloat((base + c) * 4) * x[c];
			y[r] = acc;
		});
		return y;
	}

	// ── Q4_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q4_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [d:f16(2)][dmin:f16(2)][sc:12][qs:128] = 144 bytes per 256
	 * elements. 4 groups of 64 elements; each group yields two 32-element
	 * sub-blocks via low/high nibbles of the same qs bytes — matching llama.cpp
	 * dequantize_row_q4_K.
	 *
	 * The outer row loop runs in parallel across ForkJoinPool.commonPool(). Scratch
	 * byte arrays (sc, qs) are allocated per-thread inside the lambda.
	 */
	private static float[] matVecQ4Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 144;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		int rows = rowEnd - rowStart;
		float[] y = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			// Thread-local scratch — no sharing between rows
			byte[] sc = new byte[12];
			byte[] qs = new byte[128];
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				System.arraycopy(raw, bo + 4, sc, 0, 12);
				System.arraycopy(raw, bo + 16, qs, 0, 128);

				int qi = 0;
				for (int g = 0; g < BLOCK_SIZE; g += 64) {
					int s0 = g / 32;
					int s1 = s0 + 1;
					float scale0 = d * q4kScale(sc, s0);
					float min0 = dmin * q4kMin(sc, s0);
					float scale1 = d * q4kScale(sc, s1);
					float min1 = dmin * q4kMin(sc, s1);

					for (int i = 0; i < 32; i++)
						acc += (scale0 * (qs[qi + i] & 0x0F) - min0) * x[xBase + g + i];
					for (int i = 0; i < 32; i++)
						acc += (scale1 * ((qs[qi + i] >> 4) & 0x0F) - min1) * x[xBase + g + 32 + i];
					qi += 32;
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	/**
	 * Extract 6-bit scale[j] from a Q4_K scales block (mirrors
	 * GgufReader.getScale4K).
	 */
	private static float q4kScale(byte[] sc, int j) {
		int v = (j < 4) ? sc[j] & 0x3F : ((sc[j + 4] & 0x0F) | ((sc[j - 4] & 0xC0) >> 2)) & 0x3F;
		return v;
	}

	/**
	 * Extract 6-bit min[j] from a Q4_K scales block (mirrors GgufReader.getMin4K).
	 */
	private static float q4kMin(byte[] sc, int j) {
		int v = (j < 4) ? sc[j + 4] & 0x3F : (((sc[j + 4] & 0xFF) >> 4) | ((sc[j] & 0xC0) >> 2)) & 0x3F;
		return v;
	}

	// ── Q8_0 raw bytes matVec ─────────────────────────────────────────────────

	private static float[] matVecQ8_0raw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE = 32;
		final int BLOCK_BYTES = 34; // 2 (f16 scale) + 32 (int8)
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		int rows = rowEnd - rowStart;
		float[] y = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;
			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				float scale = GgufReader.f16ToF32(readLE16(raw, bo));
				for (int i = 0; i < BLOCK_SIZE; i++)
					acc += scale * raw[bo + 2 + i] * x[xBase + i];
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	/** Read a little-endian signed 16-bit value from raw bytes at offset. */
	private static short readLE16(byte[] raw, int offset) {
		return (short) ((raw[offset] & 0xFF) | ((raw[offset + 1] & 0xFF) << 8));
	}

	// ── Q5_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q5_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [d:f16(2)][dmin:f16(2)][sc:12][qh:32][qs:128] = 176 bytes per
	 * 256 elements. 5th bit per element stored in qh: value = (nibble | (qh_bit <<
	 * 4)). Mirrors GgufReader.loadQ5_K exactly.
	 */
	private static float[] matVecQ5Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 176;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		int rows = rowEnd - rowStart;
		float[] y = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			byte[] sc = new byte[12];
			byte[] qh = new byte[32];
			byte[] qs = new byte[128];
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				float d = GgufReader.f16ToF32(readLE16(raw, bo));
				float dmin = GgufReader.f16ToF32(readLE16(raw, bo + 2));
				System.arraycopy(raw, bo + 4, sc, 0, 12);
				System.arraycopy(raw, bo + 16, qh, 0, 32);
				System.arraycopy(raw, bo + 48, qs, 0, 128);

				int qi = 0;
				for (int g = 0; g < 4; g++) {
					int s0 = g * 2;
					int s1 = s0 + 1;
					float scale0 = d * q4kScale(sc, s0);
					float min0 = dmin * q4kMin(sc, s0);
					float scale1 = d * q4kScale(sc, s1);
					float min1 = dmin * q4kMin(sc, s1);
					int hiBit0 = g * 2;
					int hiBit1 = g * 2 + 1;

					for (int l = 0; l < 32; l++) {
						int lo = qs[qi + l] & 0x0F;
						int hi = (qh[l] >>> hiBit0) & 1;
						acc += (scale0 * (lo | (hi << 4)) - min0) * x[xBase + g * 64 + l];
					}
					for (int l = 0; l < 32; l++) {
						int lo = (qs[qi + l] >>> 4) & 0x0F;
						int hi = (qh[l] >>> hiBit1) & 1;
						acc += (scale1 * (lo | (hi << 4)) - min1) * x[xBase + g * 64 + 32 + l];
					}
					qi += 32;
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	// ── Q6_K raw bytes matVec ─────────────────────────────────────────────────

	/**
	 * Q6_K block-wise matrix–vector multiply.
	 *
	 * Block layout: [ql:128][qh:64][sc:16][d:f16] = 210 bytes per 256 elements.
	 * Signed 6-bit values in [-32,31], scaled by d * sc[]. Mirrors
	 * GgufReader.loadQ6_K exactly.
	 */
	private static float[] matVecQ6Kraw(byte[] raw, float[] x, int rowStart, int rowEnd, int cols) {
		final int BLOCK_SIZE = 256;
		final int BLOCK_BYTES = 210;
		final int blocksPerRow = cols / BLOCK_SIZE;
		final int bytesPerRow = blocksPerRow * BLOCK_BYTES;
		int rows = rowEnd - rowStart;
		float[] y = new float[rows];

		java.util.stream.IntStream.range(0, rows).parallel().forEach(r -> {
			byte[] ql = new byte[128];
			byte[] qh = new byte[64];
			byte[] sc = new byte[16];
			int rowByteOffset = (rowStart + r) * bytesPerRow;
			float acc = 0f;
			int xBase = 0;

			for (int b = 0; b < blocksPerRow; b++) {
				int bo = rowByteOffset + b * BLOCK_BYTES;
				System.arraycopy(raw, bo, ql, 0, 128);
				System.arraycopy(raw, bo + 128, qh, 0, 64);
				System.arraycopy(raw, bo + 192, sc, 0, 16);
				float d = GgufReader.f16ToF32(readLE16(raw, bo + 208));

				for (int half = 0; half < 2; half++) {
					int qlBase = half * 64;
					int qhBase = half * 32;
					int scBase = half * 8;
					int xOff = xBase + half * 128;

					for (int l = 0; l < 32; l++) {
						int is = l / 16;
						int qlL = ql[qlBase + l] & 0xFF;
						int qlL2 = ql[qlBase + l + 32] & 0xFF;
						int qhL = qh[qhBase + l] & 0xFF;

						int q1 = ((qlL & 0x0F) | (((qhL >> 0) & 3) << 4)) - 32;
						int q2 = ((qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4)) - 32;
						int q3 = ((qlL >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
						int q4 = ((qlL2 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;

						float d1 = d * sc[scBase + is];
						float d2 = d * sc[scBase + is + 2];
						float d3 = d * sc[scBase + is + 4];
						float d4 = d * sc[scBase + is + 6];

						acc += d1 * q1 * x[xOff + l];
						acc += d2 * q2 * x[xOff + l + 32];
						acc += d3 * q3 * x[xOff + l + 64];
						acc += d4 * q4 * x[xOff + l + 96];
					}
				}
				xBase += BLOCK_SIZE;
			}
			y[r] = acc;
		});
		return y;
	}

	/**
	 * Rotary position embeddings (RoPE). Applied in-place to x[nHeads * headDim],
	 * treating each head independently.
	 *
	 * GGUF/llama.cpp LLaMA models use ADJACENT-pair rotation: (x[2i], x[2i+1]). The
	 * W_Q and W_K weights in the GGUF file are pre-permuted by llama.cpp's
	 * convert.py to match this convention. Using split-half pairing (x[i],
	 * x[i+headDim/2]) produces completely wrong attention scores.
	 */
	static void rope(float[] x, int pos, int nHeads, int headDim, float ropeTheta) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) Math.cos(angle);
				float sinA = (float) Math.sin(angle);
				float x0 = x[base + 2 * i];
				float x1 = x[base + 2 * i + 1];
				x[base + 2 * i] = x0 * cosA - x1 * sinA;
				x[base + 2 * i + 1] = x0 * sinA + x1 * cosA;
			}
		}
	}

	/**
	 * Grouped-query attention (GQA). q[numHeads * headDim], kCache/vCache contain
	 * the keys/values for all positions 0..seqLen-1. Returns the attended output,
	 * same shape as q.
	 */
	private float[] gqa(float[] q, float[] kCache, float[] vCache, int seqLen) {
		int H = cfg.numHeads();
		@SuppressWarnings("unused")
		int KVH = cfg.numKvHeads();
		int Hd = cfg.headDim();
		int gqa = cfg.gqaRatio();
		float scale = (float) (1.0 / Math.sqrt(Hd));
		float[] out = new float[H * Hd];
		float[] scores = new float[seqLen];

		for (int h = 0; h < H; h++) {
			int kvHead = h / gqa; // which KV head this query maps to
			int qBase = h * Hd;
			int kBase = kvHead * Hd;

			// Compute attention scores
			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				int kOffset = t * cfg.kvDim() + kBase;
				for (int d = 0; d < Hd; d++)
					dot += q[qBase + d] * kCache[kOffset + d];
				scores[t] = dot * scale;
			}

			// Softmax over scores
			softmax(scores, seqLen);

			// Weighted sum of values
			int outBase = h * Hd;
			for (int t = 0; t < seqLen; t++) {
				int vOffset = t * cfg.kvDim() + kBase;
				float w = scores[t];
				for (int d = 0; d < Hd; d++)
					out[outBase + d] += w * vCache[vOffset + d];
			}
		}
		return out;
	}

	/** In-place softmax over scores[0..n). */
	static void softmax(float[] scores, int n) {
		float max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < n; i++)
			if (scores[i] > max)
				max = scores[i];
		float sum = 0f;
		for (int i = 0; i < n; i++) {
			scores[i] = (float) Math.exp(scores[i] - max);
			sum += scores[i];
		}
		for (int i = 0; i < n; i++)
			scores[i] /= sum;
	}

	/** SiLU activation: x * sigmoid(x) */
	static float silu(float x) {
		return x / (1f + (float) Math.exp(-x));
	}

	/** Element-wise vector addition (returns new array). */
	static float[] add(float[] a, float[] b) {
		float[] out = new float[a.length];
		for (int i = 0; i < a.length; i++)
			out[i] = a[i] + b[i];
		return out;
	}
}