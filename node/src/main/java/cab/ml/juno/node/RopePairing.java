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
 * Which dimensions of a head form a rotated RoPE pair. Pair {@code i} uses the
 * same frequency in both layouts; only the pair members differ.
 *
 * <p>The right choice is fixed by how the file's Q/K weight rows are laid out,
 * not by preference: a file whose converter permuted Q/K rows for adjacent pairs
 * produces wrong attention under split-half, and the reverse. The wrong one does
 * not fail; it assigns learned frequencies to the wrong dimension pairs, and
 * quality degrades silently, more so at long context. There is deliberately no
 * runtime switch.
 */
enum RopePairing {

	/** {@code (x[2i], x[2i+1])}: files whose converter permuted Q/K rows (LLaMA). */
	ADJACENT,

	/** {@code (x[i], x[i + headDim/2])}: the rotate-half convention. */
	SPLIT_HALF
}
