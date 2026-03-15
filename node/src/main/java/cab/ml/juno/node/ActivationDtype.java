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
 * Wire encoding format for activation tensors passed between pipeline nodes.
 *
 * Choosing a smaller dtype reduces network transfer at the cost of precision
 * loss. The tradeoff is acceptable for intermediate activations — the
 * information lost in quantisation is far smaller than the noise introduced by
 * sampling.
 *
 * Compression ratios vs FLOAT32: FLOAT32 — 1× (no compression, 4 bytes/element)
 * FLOAT16 — 2× (IEEE 754 half-precision, 2 bytes/element) INT8 — ~4× (symmetric
 * quantisation, 1 byte/element + 4-byte scale header)
 *
 * Real-world impact at hidden_dim=8192, seq_len=4096 (70B model, per node-hop):
 * FLOAT32 → 64 MB → ~51 ms on 10GbE FLOAT16 → 32 MB → ~26 ms on 10GbE (2×
 * saving) INT8 → 16 MB → ~13 ms on 10GbE (~4× saving)
 *
 * Recommendation: FLOAT16 for most pipelines — excellent compression with
 * negligible accuracy loss. INT8 for bandwidth-constrained community nodes.
 */
public enum ActivationDtype {

	/**
	 * IEEE 754 single-precision (32-bit) — lossless, 4 bytes/element. Default. Use
	 * when accuracy is paramount or network is not a bottleneck.
	 */
	FLOAT32,

	/**
	 * IEEE 754 half-precision (16-bit) — 2× compression, bounded precision loss.
	 * Max representable value: 65504. Well within typical normalised activation
	 * range. Relative error: ~0.1% for values in [-1, 1].
	 */
	FLOAT16,

	/**
	 * Symmetric INT8 quantisation — ~4× compression, higher precision loss. Wire
	 * layout:
	 * {@code [scale:float32 big-endian (4 bytes)][quantised:signed byte × N]}.
	 * Scale = max(|activations|) / 127. Quantise = round(f / scale), clamped to
	 * [−127, 127]. Reconstruct = byte × scale.
	 */
	INT8
}