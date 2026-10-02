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
 * The output side of {@link MatVec}'s non-allocating batched form
 * ({@code sgemmInto}): the argument check every implementation runs before
 * computing, and the row copy the interface defaults use.
 */
final class SgemmOutput {

	private SgemmOutput() {
	}

	/**
	 * Checks that {@code Y} can take {@code X.length} result rows of {@code rows}
	 * values each.
	 *
	 * @throws IllegalArgumentException when {@code Y} has fewer rows than {@code X},
	 *                                  or a row is null or shorter than {@code rows}
	 */
	static void require(float[][] X, float[][] Y, int rows) {
		if (Y == null || Y.length < X.length)
			throw new IllegalArgumentException(
					"output has " + (Y == null ? "no" : Y.length) + " rows for a batch of " + X.length);
		for (int b = 0; b < X.length; b++) {
			if (Y[b] == null || Y[b].length < rows)
				throw new IllegalArgumentException("output row " + b + " is "
						+ (Y[b] == null ? "null" : Y[b].length + " long") + ", the result is " + rows + " wide");
		}
	}

	/** Copies the first {@code rows} values of every row of {@code from} into {@code Y}. */
	static void copy(float[][] from, float[][] Y, int rows) {
		for (int b = 0; b < from.length; b++)
			System.arraycopy(from[b], 0, Y[b], 0, rows);
	}
}
