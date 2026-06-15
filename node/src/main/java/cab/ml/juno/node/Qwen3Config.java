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

/**
 * Qwen3 / Qwen3-MoE model configuration from GGUF metadata.
 *
 * <p>Extends {@link LlamaConfig} with {@code attention.key_length} for head_dim and
 * optional MoE fields ({@code expert_count}, {@code expert_used_count},
 * {@code expert_feed_forward_length}).
 */
public record Qwen3Config(
		LlamaConfig base,
		int expertCount,
		int expertUsedCount,
		int expertFeedForwardLength,
		float expertWeightsScale,
		boolean expertWeightsNorm,
		Qwen3RopeConfig rope) {

	public static Qwen3Config from(GgufReader r) throws IOException {
		LlamaConfig raw = LlamaConfig.from(r);
		String arch = raw.architecture();
		String p = arch + ".";

		int headDim = r.metaInt(p + "attention.key_length", raw.headDim());
		if (headDim <= 0)
			headDim = raw.hiddenDim() / raw.numHeads();

		LlamaConfig base = new LlamaConfig(raw.hiddenDim(), raw.numLayers(), raw.numHeads(), raw.numKvHeads(), headDim,
				raw.intermediateSize(), raw.vocabSize(), raw.rmsNormEps(), raw.ropeTheta(), arch);

		boolean moe = "qwen3moe".equalsIgnoreCase(arch);
		int expertCount = moe ? r.metaInt(p + "expert_count", 0) : 0;
		int expertUsed = moe ? r.metaInt(p + "expert_used_count", 0) : 0;
		int expertFf = moe ? r.metaInt(p + "expert_feed_forward_length", 0) : 0;
		if (moe && expertFf <= 0)
			expertFf = base.intermediateSize();
		float expertScale = moe ? r.metaFloat(p + "expert_weights_scale", 1.0f) : 1.0f;
		boolean expertNorm = moe && r.metaInt(p + "expert_weights_norm", 0) != 0;

		Qwen3RopeConfig rope = Qwen3RopeConfig.from(r, base);
		return new Qwen3Config(base, expertCount, expertUsed, expertFf, expertScale, expertNorm, rope);
	}

	public int hiddenDim() {
		return base.hiddenDim();
	}

	public int numLayers() {
		return base.numLayers();
	}

	public int numHeads() {
		return base.numHeads();
	}

	public int numKvHeads() {
		return base.numKvHeads();
	}

	public int headDim() {
		return base.headDim();
	}

	public int intermediateSize() {
		return base.intermediateSize();
	}

	public int vocabSize() {
		return base.vocabSize();
	}

	public float rmsNormEps() {
		return base.rmsNormEps();
	}

	public float ropeTheta() {
		return base.ropeTheta();
	}

	public String architecture() {
		return base.architecture();
	}

	public int gqaRatio() {
		return base.gqaRatio();
	}

	public int kvDim() {
		return base.kvDim();
	}

	/** Query projection width = numHeads × headDim (may exceed hiddenDim). */
	public int qDim() {
		return numHeads() * headDim();
	}

	public boolean isMoe() {
		return expertCount > 0;
	}

	@Override
	public String toString() {
		if (isMoe()) {
			return String.format(
					"Qwen3Config{arch=%s hidden=%d layers=%d heads=%d kvHeads=%d headDim=%d ffn=%d vocab=%d"
							+ " experts=%d used=%d expertFf=%d}",
					architecture(), hiddenDim(), numLayers(), numHeads(), numKvHeads(), headDim(), intermediateSize(),
					vocabSize(), expertCount, expertUsedCount, expertFeedForwardLength);
		}
		return String.format(
				"Qwen3Config{arch=%s hidden=%d layers=%d heads=%d kvHeads=%d headDim=%d ffn=%d vocab=%d yarn=%s}",
				architecture(), hiddenDim(), numLayers(), numHeads(), numKvHeads(), headDim(), intermediateSize(),
				vocabSize(), rope.yarn());
	}
}
