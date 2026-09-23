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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Combines one or more {@code --lora-play} adapter sets, each carrying its own
 * playback scale, into a single {@link LoraAdapterSet} that the existing
 * single-set inference plumbing ({@code ForwardPassHandlerLoader},
 * {@code LoraTrainableHandler}, ...) can consume unchanged.
 *
 * <p>
 * For a (layer, projection) key present in several scaled sets, the desired
 * forward-pass delta is {@code sum_i scale_i * adapter_i.forward(x)}, where
 * {@code adapter_i.forward(x) = adapter_i.scale * B_i * (A_i * x)} already
 * folds in that adapter's own alpha/rank scaling. This is reproduced exactly,
 * without threading a list through every forward-pass call site, by
 * rank-concatenating the entries: stack every {@code A_i} as extra rows
 * (their own {@code inDim} is unchanged) and every {@code B_i} as extra
 * columns pre-multiplied by {@code scale_i * adapter_i.scale}, then wrap the
 * concatenation in one merged adapter whose own scale is fixed at exactly
 * {@code 1.0} (alpha == rank, standard scaling) so it re-applies no further
 * scaling. Matrix multiplication is linear, so
 * {@code B_merged * (A_merged * x) == sum_i (scale_i * adapter_i.scale) * B_i * (A_i * x)}.
 *
 * <p>
 * QA-LoRA and DoRA entries are not supported by this merge yet — combining
 * their grouped/magnitude-adjusted math correctly needs its own derivation,
 * so a set containing either fails closed rather than silently dropping the
 * scale or magnitude term. Use a single plain adapter file at scale 1.0 for
 * those checkpoints until a follow-up tier adds support.
 */
public final class LoraPlaybackMerge {

	/** One {@code --lora-play} entry: an adapter file plus its playback scale. */
	public record ScaledAdapterSet(LoraAdapterSet adapters, float scale, String sourceLabel) {
		public ScaledAdapterSet {
			Objects.requireNonNull(adapters, "adapters");
			Objects.requireNonNull(sourceLabel, "sourceLabel");
			if (!Float.isFinite(scale))
				throw new IllegalArgumentException("scale must be finite: " + scale);
		}
	}

	private record Weighted(LoraAdapter adapter, float playScale, String sourceLabel) {
	}

	private LoraPlaybackMerge() {
	}

	/**
	 * Merge {@code sets} into one playback-ready {@link LoraAdapterSet}. The
	 * single-set, scale-1.0 case returns the original set unchanged (no format
	 * perturbation) so the original single-file {@code --lora-play file.lora}
	 * path stays byte-for-byte back-compatible.
	 */
	public static LoraAdapterSet merge(List<ScaledAdapterSet> sets) {
		if (sets == null || sets.isEmpty())
			throw new IllegalArgumentException("at least one LoRA adapter set is required");
		if (sets.size() == 1 && sets.get(0).scale() == 1.0f)
			return sets.get(0).adapters();

		for (ScaledAdapterSet s : sets) {
			if (!s.adapters().allQa().isEmpty())
				throw new IllegalArgumentException(
						"multi-adapter / scaled --lora-play does not support QA-LoRA adapters yet (" + s.sourceLabel()
								+ "); use a single adapter file at scale 1.0");
			if (!s.adapters().magnitudes().isEmpty())
				throw new IllegalArgumentException(
						"multi-adapter / scaled --lora-play does not support DoRA adapters yet (" + s.sourceLabel()
								+ "); use a single adapter file at scale 1.0");
		}

		Map<String, List<Weighted>> byKey = new LinkedHashMap<>();
		for (ScaledAdapterSet s : sets) {
			for (Map.Entry<String, LoraAdapter> e : s.adapters().asMap().entrySet()) {
				byKey.computeIfAbsent(e.getKey(), k -> new ArrayList<>())
						.add(new Weighted(e.getValue(), s.scale(), s.sourceLabel()));
			}
		}

		LoraAdapterSet merged = new LoraAdapterSet();
		for (Map.Entry<String, List<Weighted>> e : byKey.entrySet()) {
			String key = e.getKey();
			merged.add(LoraAdapterSet.keyLayer(key), LoraAdapterSet.keyProj(key), mergeEntries(key, e.getValue()));
		}
		return merged;
	}

	private static LoraAdapter mergeEntries(String key, List<Weighted> entries) {
		int inDim = entries.get(0).adapter().inDim;
		int outDim = entries.get(0).adapter().outDim;
		for (Weighted w : entries) {
			if (w.adapter().inDim != inDim || w.adapter().outDim != outDim)
				throw new IllegalArgumentException("shape mismatch at " + key + ": " + w.sourceLabel() + " is "
						+ w.adapter().outDim + "x" + w.adapter().inDim + ", expected " + outDim + "x" + inDim);
		}

		int totalRank = 0;
		for (Weighted w : entries)
			totalRank += w.adapter().rank;

		float[] aMerged = new float[totalRank * inDim];
		float[] bMerged = new float[outDim * totalRank];
		int rankOffset = 0;
		int aOffset = 0;
		for (Weighted w : entries) {
			LoraAdapter adapter = w.adapter();
			float folded = w.playScale() * adapter.scale;

			float[] ai = adapter.a();
			System.arraycopy(ai, 0, aMerged, aOffset, ai.length);
			aOffset += ai.length;

			float[] bi = adapter.b();
			for (int r = 0; r < outDim; r++) {
				int srcBase = r * adapter.rank;
				int dstBase = r * totalRank + rankOffset;
				for (int c = 0; c < adapter.rank; c++)
					bMerged[dstBase + c] = folded * bi[srcBase + c];
			}
			rankOffset += adapter.rank;
		}

		// alpha == rank under standard scaling gives effectiveScale() == 1.0
		// exactly, since every per-adapter scale is already folded into bMerged.
		LoraAdapterConfig config = LoraAdapterConfig.of(totalRank, (float) totalRank);
		return LoraAdapter.fromWeights(config, inDim, outDim, aMerged, bMerged);
	}
}
