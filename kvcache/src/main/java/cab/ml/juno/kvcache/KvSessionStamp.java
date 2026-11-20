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
package cab.ml.juno.kvcache;

import java.util.Arrays;

/**
 * What a saved session's KV belongs to, written at the head of each stage's file and checked again on restore: the
 * model file's identity, the LoRA adapter's (empty without one), the stage that holds these layers (its layer range,
 * and its rank on a tensor-parallel node), and the tokens whose positions the rows hold.
 */
public record KvSessionStamp(String model, String adapter, String stage, int[] tokens) {

	public KvSessionStamp {
		if (model == null || model.isBlank())
			throw new IllegalArgumentException("model must not be blank");
		if (adapter == null)
			throw new IllegalArgumentException("adapter must not be null (empty without one)");
		if (stage == null || stage.isBlank())
			throw new IllegalArgumentException("stage must not be blank");
		if (tokens == null || tokens.length < 1)
			throw new IllegalArgumentException("tokens must hold at least one token");
		tokens = tokens.clone();
	}

	@Override
	public int[] tokens() {
		return tokens.clone();
	}

	/** Number of positions the rows hold. */
	public int positions() {
		return tokens.length;
	}

	/** This stamp for another stage of the same save. */
	public KvSessionStamp forStage(String otherStage) {
		return new KvSessionStamp(model, adapter, otherStage, tokens);
	}

	@Override
	public boolean equals(Object o) {
		return o instanceof KvSessionStamp s && model.equals(s.model) && adapter.equals(s.adapter)
				&& stage.equals(s.stage) && Arrays.equals(tokens, s.tokens);
	}

	@Override
	public int hashCode() {
		return 31 * (31 * (31 * model.hashCode() + adapter.hashCode()) + stage.hashCode()) + Arrays.hashCode(tokens);
	}

	@Override
	public String toString() {
		return "KvSessionStamp[model=" + model + ", adapter=" + adapter + ", stage=" + stage + ", positions="
				+ tokens.length + "]";
	}
}
