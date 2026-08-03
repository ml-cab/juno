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

/**
 * CPU reference for microbatched frozen linears (Tier 9 parity oracle).
 *
 * <p>Column layout matches {@link DeviceActivationBatch#packColumns}: each
 * {@code X[b]} / {@code G[b]} is one activation / gradient column.
 */
final class CpuFrozenBatchOps {

	private CpuFrozenBatchOps() {
	}

	/** {@code Y[b] = W * X[b]} for row-major {@code W[rows×cols]}. */
	static float[][] forward(float[] W, float[][] X, int rows, int cols) {
		int batch = X.length;
		float[][] Y = new float[batch][];
		for (int b = 0; b < batch; b++) {
			float[] x = X[b];
			if (x.length != cols)
				throw new IllegalArgumentException("X[" + b + "].length != cols");
			float[] y = new float[rows];
			for (int r = 0; r < rows; r++) {
				float acc = 0f;
				int base = r * cols;
				for (int c = 0; c < cols; c++)
					acc += W[base + c] * x[c];
				y[r] = acc;
			}
			Y[b] = y;
		}
		return Y;
	}

	/** {@code dX[b] = W^T * G[b]} for row-major {@code W[rows×cols]}. */
	static float[][] transpose(float[] W, float[][] G, int rows, int cols) {
		int batch = G.length;
		float[][] dX = new float[batch][];
		for (int b = 0; b < batch; b++) {
			float[] g = G[b];
			if (g.length != rows)
				throw new IllegalArgumentException("G[" + b + "].length != rows");
			float[] z = new float[cols];
			for (int c = 0; c < cols; c++) {
				float acc = 0f;
				for (int r = 0; r < rows; r++)
					acc += W[r * cols + c] * g[r];
				z[c] = acc;
			}
			dX[b] = z;
		}
		return dX;
	}
}
