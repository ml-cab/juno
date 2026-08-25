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
package cab.ml.juno.coordinator;

import java.util.Locale;

/**
 * Prefill strategy for {@link GenerationLoop}.
 *
 * <ul>
 *   <li>{@link #BATCHED} (default): processes all new prompt tokens in one
 *       {@link cab.ml.juno.node.InferencePipeline#prefillBatch} call, which
 *       {@link cab.ml.juno.node.LocalInferencePipeline} implements as a single
 *       batched GEMM pass through the handler chain — removing the O(N&sup2;)
 *       {@code copyOfRange} churn and enabling one matmul per weight matrix per
 *       layer instead of N GEMVs.</li>
 *   <li>{@link #SINGLE}: the original sequential one-token-at-a-time loop, kept
 *       verbatim as a permanent escape hatch for bisection, GPU-vendor bug
 *       workarounds, or like-for-like comparison against performance baselines.
 *       No rebuild or code change required to fall back to it — just pass
 *       {@code --prefill single}.</li>
 * </ul>
 *
 * Controlled by the {@code --prefill single|batched} CLI flag in
 * {@code ConsoleMain}. Default when the flag is absent: {@link #BATCHED}.
 */
public enum PrefillMode {
	/**
	 * Today's sequential one-token-at-a-time prefill loop (legacy / escape
	 * hatch).
	 */
	SINGLE,

	/**
	 * New windowed GEMM prefill — processes all new prompt tokens in one batched
	 * call. Default.
	 */
	BATCHED;

	/**
	 * Parse a CLI string to a {@link PrefillMode}, case-insensitive.
	 *
	 * @param s the raw value from {@code --prefill} (e.g. {@code "single"},
	 *          {@code "BATCHED"})
	 * @return the corresponding mode
	 * @throws IllegalArgumentException if {@code s} is not a recognised value
	 */
	public static PrefillMode parse(String s) {
		return switch (s.toLowerCase(Locale.ROOT)) {
		case "single"  -> SINGLE;
		case "batched" -> BATCHED;
		default -> throw new IllegalArgumentException(
				"Unrecognized --prefill value '" + s + "' (expected: single, batched)");
		};
	}
}
