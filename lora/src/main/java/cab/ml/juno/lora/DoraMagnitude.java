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
 * DoRA magnitude vector and its gradient accumulator (length = outDim).
 */
public final class DoraMagnitude {

	private final float[] values;
	private final float[] grad;

	public DoraMagnitude(int outDim) {
		if (outDim < 1)
			throw new IllegalArgumentException("outDim must be >= 1");
		this.values = new float[outDim];
		this.grad = new float[outDim];
	}

	public static DoraMagnitude fromValues(float[] values) {
		Objects.requireNonNull(values, "values");
		requireFinite(values, "magnitude");
		DoraMagnitude m = new DoraMagnitude(values.length);
		System.arraycopy(values, 0, m.values, 0, values.length);
		return m;
	}

	public int length() {
		return values.length;
	}

	public float[] values() {
		return values;
	}

	public float[] grad() {
		return grad;
	}

	public void zeroGrad() {
		Arrays.fill(grad, 0f);
	}

	public void copyFrom(DoraMagnitude src) {
		if (src.length() != length())
			throw new IllegalArgumentException("magnitude length mismatch");
		System.arraycopy(src.values, 0, values, 0, values.length);
		zeroGrad();
	}

	public DoraMagnitude copy() {
		return fromValues(values);
	}

	private static void requireFinite(float[] arr, String label) {
		for (float f : arr) {
			if (!Float.isFinite(f))
				throw new IllegalArgumentException("non-finite " + label);
		}
	}
}
