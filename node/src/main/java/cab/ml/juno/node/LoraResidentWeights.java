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

import java.util.logging.Logger;

/**
 * Shared GPU residency helpers for LoRA training handlers (Tier 10 / Tier 9).
 *
 * <p>Uploads dequantized (or dense host) projections to {@link ResidentWeightMatrix},
 * closes partial uploads on VRAM failure, and routes frozen forward / transpose
 * through resident {@code sgemv} / {@code sgemvTranspose} when present.
 *
 * <p>When {@link #microbatchSize()} {@code > 1}, uploads prefer FP32 so
 * {@link GpuBlasOps} microbatched GEMM can run. Honors {@code juno.lora.train.device}:
 * {@code gpu} fails closed on upload OOM; {@code auto} logs and falls back to CPU.
 */
final class LoraResidentWeights {

	/** Default microbatch width for frozen GEMM (Tier 9). Override with {@code juno.lora.microbatch}. */
	static final int DEFAULT_MICROBATCH = 8;

	private LoraResidentWeights() {
	}

	/**
	 * Microbatch size for frozen linears. {@code 1} disables GEMM batching
	 * (sequential GEMV). Default {@value #DEFAULT_MICROBATCH}.
	 */
	static int microbatchSize() {
		int n = Integer.getInteger("juno.lora.microbatch", DEFAULT_MICROBATCH);
		return Math.max(1, n);
	}

	/** Upload host row-major matrix; FP32 when microbatching, else FP16 when supported. */
	static ResidentWeightMatrix upload(GpuMatVec gpu, float[] host, int rows, int cols) {
		if (microbatchSize() > 1)
			return ResidentWeightMatrix.uploadFp32(gpu, host, rows, cols);
		return gpu.supportsHalfResident()
				? ResidentWeightMatrix.uploadHalf(gpu, host, rows, cols)
				: ResidentWeightMatrix.uploadFp32(gpu, host, rows, cols);
	}

	/** Dequantize then upload a GGUF projection. */
	static ResidentWeightMatrix uploadQuant(GpuMatVec gpu, GgufReader.QuantizedTensor t, int rows, int cols) {
		return upload(gpu, LlamaTransformerHandler.dequantize(t, rows, cols), rows, cols);
	}

	static void closeQuietly(ResidentWeightMatrix m) {
		if (m != null && !m.isClosed())
			m.close();
	}

	static void closeArray(ResidentWeightMatrix[] a) {
		if (a == null)
			return;
		for (ResidentWeightMatrix m : a)
			closeQuietly(m);
	}

	static boolean isVramOom(IllegalStateException ex) {
		String msg = ex.getMessage();
		return msg != null && (msg.contains("cudaMalloc") || msg.contains("hipMalloc"));
	}

	/**
	 * Close partial uploads after a failed upload attempt.
	 *
	 * @return {@code true} if the caller should continue on CPU (VRAM OOM under auto)
	 * @throws IllegalStateException when {@code --lora-train-device=gpu} and VRAM OOM,
	 *         or when the error is not a VRAM allocation failure
	 */
	static boolean tryRecoverFromUploadOom(IllegalStateException ex, Logger log, Runnable closer) {
		closer.run();
		if (!isVramOom(ex))
			throw ex;
		String mode = System.getProperty("juno.lora.train.device", LoraTrainDevice.AUTO);
		if (LoraTrainDevice.requireResident(mode)) {
			throw new IllegalStateException(
					"--lora-train-device=gpu: insufficient GPU VRAM for resident weights (" + ex.getMessage() + ")",
					ex);
		}
		log.warning("LoRA: insufficient GPU VRAM for resident weights (" + ex.getMessage()
				+ "). Using CPU quantised matmul.");
		return true;
	}

	/** Frozen forward {@code W*x}: resident GPU when {@code dev != null}, else quantized CPU. */
	static float[] matVec(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, float[] x, int rows, int cols) {
		if (dev != null)
			return dev.sgemv(x);
		return LlamaTransformerHandler.matVec(quant, x, rows, cols);
	}

	/** Frozen transpose {@code W^T*g}: resident GPU when {@code dev != null}, else quantized CPU. */
	static float[] transposedMatVec(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, float[] g, int rows,
			int cols) {
		if (dev != null)
			return dev.sgemvTranspose(g);
		return LoraTrainableHandler.transposedMatVec(quant, g, rows, cols);
	}

	/**
	 * Microbatched frozen forward. Uses {@link GpuBlasOps} when {@code dev} is FP32-resident;
	 * otherwise sequential {@link #matVec}. Processes in windows of {@link #microbatchSize()}.
	 */
	static float[][] matVecBatch(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, GpuBlasOps ops,
			float[][] X, int batch, int rows, int cols) {
		if (batch <= 0)
			return new float[0][];
		if (dev == null || ops == null || !dev.supportsBatchedSgemm() || microbatchSize() <= 1) {
			float[][] Y = new float[batch][];
			for (int b = 0; b < batch; b++)
				Y[b] = matVec(quant, dev, X[b], rows, cols);
			return Y;
		}
		int mb = microbatchSize();
		float[][] Y = new float[batch][];
		for (int start = 0; start < batch; start += mb) {
			int n = Math.min(mb, batch - start);
			float[][] slice = new float[n][];
			System.arraycopy(X, start, slice, 0, n);
			float[][] out = dev.sgemmBatch(ops, slice, n);
			System.arraycopy(out, 0, Y, start, n);
		}
		return Y;
	}

	/**
	 * Microbatched frozen transpose. Same fallback policy as {@link #matVecBatch}.
	 */
	static float[][] transposedMatVecBatch(GgufReader.QuantizedTensor quant, ResidentWeightMatrix dev, GpuBlasOps ops,
			float[][] G, int batch, int rows, int cols) {
		if (batch <= 0)
			return new float[0][];
		if (dev == null || ops == null || !dev.supportsBatchedSgemm() || microbatchSize() <= 1) {
			float[][] dX = new float[batch][];
			for (int b = 0; b < batch; b++)
				dX[b] = transposedMatVec(quant, dev, G[b], rows, cols);
			return dX;
		}
		int mb = microbatchSize();
		float[][] dX = new float[batch][];
		for (int start = 0; start < batch; start += mb) {
			int n = Math.min(mb, batch - start);
			float[][] slice = new float[n][];
			System.arraycopy(G, start, slice, 0, n);
			float[][] out = dev.sgemmTransposeBatch(ops, slice, n);
			System.arraycopy(out, 0, dX, start, n);
		}
		return dX;
	}

	/** Dense host forward {@code W*x} (Phi-3 / Qwen3 output projection). */
	static float[] matVecDense(float[] A, ResidentWeightMatrix dev, float[] x, int rows, int cols) {
		if (dev != null)
			return dev.sgemv(x);
		return LlamaTransformerHandler.matVec(A, x, rows, cols);
	}

	/** Dense host transpose {@code W^T*g}. */
	static float[] transposedMatVecDense(float[] A, ResidentWeightMatrix dev, float[] g, int rows, int cols) {
		if (dev != null)
			return dev.sgemvTranspose(g);
		float[] y = new float[cols];
		for (int c = 0; c < cols; c++) {
			float s = 0f;
			for (int r = 0; r < rows; r++)
				s += A[r * cols + c] * g[r];
			y[c] = s;
		}
		return y;
	}

	/** Contiguous row block {@code A[rowStart : rowStart+nRows, 0:cols]} in row-major {@code full}. */
	static float[] rowMajorSlice(float[] full, int rowStart, int nRows, int cols) {
		float[] out = new float[nRows * cols];
		System.arraycopy(full, rowStart * cols, out, 0, nRows * cols);
		return out;
	}
}
