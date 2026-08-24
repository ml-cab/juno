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
package cab.ml.juno.lora;

import java.util.Arrays;
import java.util.Objects;

/**
 * Canonical detached-norm DoRA for one dense projection.
 *
 * <pre>
 *   direction = W + scale·B·A
 *   coefficient = magnitude / max(‖direction‖_row, ε)   // norm detached
 *   output = coefficient ⊙ (W·x + scale·B·A·x)
 * </pre>
 *
 * Row norms are refreshed from the dense F32 base {@code W} plus the current
 * LoRA direction. Call {@link #markDirty()} after optimizer updates (or any
 * direct A/B/magnitude mutation); {@link #forward} refreshes once when dirty.
 */
public final class DoraProjection {

	public static final float EPS = 1e-6f;

	private final float[] w;
	private final LoraAdapter lora;
	private final DoraMagnitude magnitude;
	private final float[] rowNorms;
	private final float[] coefficients;
	private boolean dirty = true;

	/** Last forward direction output {@code W·x + Δ} (length outDim). */
	private float[] lastDirectionOut;
	/** Coefficient snapshot used for the last forward (length outDim). */
	private float[] lastCoefficients;
	/** Norm snapshot used for the last forward (length outDim). */
	private float[] lastNorms;

	/**
	 * @param w          dense frozen weights, row-major {@code outDim × inDim}
	 * @param lora       low-rank adapter (must be {@link LoraMode#DORA})
	 * @param magnitude  length {@code outDim}; typically initialised from row norms of {@code W}
	 */
	public DoraProjection(float[] w, LoraAdapter lora, DoraMagnitude magnitude) {
		this.w = Objects.requireNonNull(w, "w");
		this.lora = Objects.requireNonNull(lora, "lora");
		this.magnitude = Objects.requireNonNull(magnitude, "magnitude");
		if (lora.mode != LoraMode.DORA)
			throw new IllegalArgumentException("adapter mode must be DORA");
		if (w.length != lora.outDim * lora.inDim)
			throw new IllegalArgumentException("W length mismatch");
		if (magnitude.length() != lora.outDim)
			throw new IllegalArgumentException("magnitude length mismatch");
		this.rowNorms = new float[lora.outDim];
		this.coefficients = new float[lora.outDim];
	}

	public LoraAdapter adapter() {
		return lora;
	}

	public DoraMagnitude magnitude() {
		return magnitude;
	}

	public float[] rowNorms() {
		ensureFresh();
		return rowNorms;
	}

	public float[] coefficients() {
		ensureFresh();
		return coefficients;
	}

	public void markDirty() {
		dirty = true;
	}

	public boolean dirty() {
		return dirty;
	}

	/**
	 * Initialise magnitude from exact row norms of {@code W} so B=0 reproduces the
	 * base matvec.
	 */
	public static DoraMagnitude magnitudeFromBaseRows(float[] w, int outDim, int inDim) {
		if (w.length != outDim * inDim)
			throw new IllegalArgumentException("W length mismatch");
		float[] mag = new float[outDim];
		for (int r = 0; r < outDim; r++) {
			double sum = 0;
			int base = r * inDim;
			for (int c = 0; c < inDim; c++) {
				float v = w[base + c];
				sum += (double) v * v;
			}
			mag[r] = (float) Math.sqrt(sum);
		}
		return DoraMagnitude.fromValues(mag);
	}

	/** Refresh row norms and coefficients from current W + LoRA direction. */
	public void refresh() {
		int out = lora.outDim;
		int in = lora.inDim;
		int rank = lora.rank;
		float scale = lora.scale;
		float[] a = lora.a();
		float[] b = lora.b();
		float[] mag = magnitude.values();

		for (int r = 0; r < out; r++) {
			int wBase = r * in;
			int bBase = r * rank;
			double sumSq = 0;
			for (int c = 0; c < in; c++) {
				float dir = w[wBase + c];
				for (int k = 0; k < rank; k++)
					dir += scale * b[bBase + k] * a[k * in + c];
				sumSq += (double) dir * dir;
			}
			float norm = (float) Math.sqrt(sumSq);
			rowNorms[r] = norm;
			float denom = Math.max(norm, EPS);
			coefficients[r] = mag[r] / denom;
		}
		dirty = false;
	}

	/**
	 * DoRA forward: {@code y = coeff ⊙ (W·x + LoRA(x))}.
	 */
	public float[] forward(float[] x) {
		ensureFresh();
		int out = lora.outDim;
		int in = lora.inDim;
		if (x.length != in)
			throw new IllegalArgumentException("x length mismatch");

		float[] directionOut = new float[out];
		for (int r = 0; r < out; r++) {
			float acc = 0f;
			int wBase = r * in;
			for (int c = 0; c < in; c++)
				acc += w[wBase + c] * x[c];
			directionOut[r] = acc;
		}
		float[] delta = lora.forward(x);
		for (int r = 0; r < out; r++)
			directionOut[r] += delta[r];

		float[] y = new float[out];
		for (int r = 0; r < out; r++)
			y[r] = coefficients[r] * directionOut[r];

		lastDirectionOut = directionOut;
		lastCoefficients = Arrays.copyOf(coefficients, out);
		lastNorms = Arrays.copyOf(rowNorms, out);
		return y;
	}

	/**
	 * DoRA backward with detached row norms.
	 *
	 * <ul>
	 * <li>{@code gradMagnitude[r] += gradOut[r] * directionOut[r] / norm[r]}
	 * <li>LoRA and frozen-W receive {@code scaledGrad = coeff ⊙ gradOut}
	 * <li>Returns {@code gradX = Wᵀ·scaledGrad + LoRA.backward(scaledGrad, x)}
	 * </ul>
	 */
	public float[] backward(float[] gradOut, float[] x) {
		if (lastDirectionOut == null || lastCoefficients == null || lastNorms == null)
			throw new IllegalStateException("backward requires a prior forward");
		int out = lora.outDim;
		int in = lora.inDim;
		if (gradOut.length != out || x.length != in)
			throw new IllegalArgumentException("shape mismatch");

		float[] scaledGrad = new float[out];
		float[] magGrad = magnitude.grad();
		for (int r = 0; r < out; r++) {
			float g = gradOut[r];
			scaledGrad[r] = lastCoefficients[r] * g;
			float norm = lastNorms[r];
			float denom = Math.max(norm, EPS);
			magGrad[r] += g * lastDirectionOut[r] / denom;
		}

		float[] gradXLora = lora.backward(scaledGrad, x);
		float[] gradX = new float[in];
		for (int c = 0; c < in; c++) {
			float acc = 0f;
			for (int r = 0; r < out; r++)
				acc += w[r * in + c] * scaledGrad[r];
			gradX[c] = acc + gradXLora[c];
		}
		return gradX;
	}

	/**
	 * Scale an already-computed base+delta vector by cached coefficients (handler
	 * fast path when {@code W·x} was computed elsewhere). Stores backward state.
	 */
	public float[] scaleDirectionOutput(float[] directionOut) {
		ensureFresh();
		int out = lora.outDim;
		if (directionOut.length != out)
			throw new IllegalArgumentException("directionOut length mismatch");
		float[] y = new float[out];
		for (int r = 0; r < out; r++)
			y[r] = coefficients[r] * directionOut[r];
		lastDirectionOut = directionOut.clone();
		lastCoefficients = Arrays.copyOf(coefficients, out);
		lastNorms = Arrays.copyOf(rowNorms, out);
		return y;
	}

	/**
	 * Accumulate magnitude grads and return {@code coeff ⊙ gradOut} for frozen
	 * transpose + LoRA backward (handler path after {@link #scaleDirectionOutput}).
	 */
	public float[] scaleGradient(float[] gradOut) {
		if (lastDirectionOut == null || lastCoefficients == null || lastNorms == null)
			throw new IllegalStateException("scaleGradient requires a prior DoRA forward");
		int out = lora.outDim;
		if (gradOut.length != out)
			throw new IllegalArgumentException("gradOut length mismatch");
		float[] scaled = new float[out];
		float[] magGrad = magnitude.grad();
		for (int r = 0; r < out; r++) {
			float g = gradOut[r];
			scaled[r] = lastCoefficients[r] * g;
			float denom = Math.max(lastNorms[r], EPS);
			magGrad[r] += g * lastDirectionOut[r] / denom;
		}
		return scaled;
	}

	private void ensureFresh() {
		if (dirty)
			refresh();
	}
}
