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

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Canonical LoRA projection metadata: external key, GGUF tensor suffix, and
 * dimensions relative to {@link LlamaConfig}.
 *
 * <p>
 * External keys are lowercase and stable in CLI/checkpoints:
 * {@code wq,wk,wv,wo,wgate,wup,wdown}.
 */
public enum LoraProjection {

	WQ("wq", "attn_q.weight"),
	WK("wk", "attn_k.weight"),
	WV("wv", "attn_v.weight"),
	WO("wo", "attn_output.weight"),
	WGATE("wgate", "ffn_gate.weight"),
	WUP("wup", "ffn_up.weight"),
	WDOWN("wdown", "ffn_down.weight");

	private final String key;
	private final String ggufSuffix;

	LoraProjection(String key, String ggufSuffix) {
		this.key = key;
		this.ggufSuffix = ggufSuffix;
	}

	/** Lowercase external key used in checkpoints and CLI. */
	public String key() {
		return key;
	}

	/** GGUF tensor name suffix after {@code blk.L.}. */
	public String ggufSuffix() {
		return ggufSuffix;
	}

	/** Full GGUF tensor name for the given absolute layer index. */
	public String ggufTensorName(int layer) {
		return "blk." + layer + "." + ggufSuffix;
	}

	public int inDim(LlamaConfig cfg) {
		return switch (this) {
		case WDOWN -> cfg.intermediateSize();
		default -> cfg.hiddenDim();
		};
	}

	public int outDim(LlamaConfig cfg) {
		return switch (this) {
		case WK, WV -> cfg.kvDim();
		case WGATE, WUP -> cfg.intermediateSize();
		default -> cfg.hiddenDim();
		};
	}

	public static LoraProjection fromKey(String key) {
		if (key == null || key.isBlank())
			throw new IllegalArgumentException("empty projection key");
		String k = key.strip().toLowerCase(Locale.ROOT);
		for (LoraProjection p : values()) {
			if (p.key.equals(k))
				return p;
		}
		throw new IllegalArgumentException("unknown LoRA projection: " + key);
	}

	/** Default training targets: query and value projections. */
	public static List<LoraProjection> qv() {
		return List.of(WQ, WV);
	}

	/** All seven dense linear projections in stable order. */
	public static List<LoraProjection> allLinear() {
		return List.of(values());
	}

	/**
	 * Parse a target specification.
	 *
	 * <ul>
	 * <li>{@code qv} — query and value
	 * <li>{@code all} / {@code all-linear} — every supported linear projection
	 * <li>comma-separated keys — explicit ordered unique set
	 * </ul>
	 *
	 * @throws IllegalArgumentException on unknown, duplicate, or empty sets
	 */
	public static List<LoraProjection> parseTargets(String spec) {
		if (spec == null || spec.isBlank())
			throw new IllegalArgumentException("LoRA targets must not be empty");
		String s = spec.strip().toLowerCase(Locale.ROOT);
		if (s.equals("qv"))
			return qv();
		if (s.equals("all") || s.equals("all-linear"))
			return allLinear();

		String[] parts = s.split(",", -1);
		Set<LoraProjection> seen = EnumSet.noneOf(LoraProjection.class);
		List<LoraProjection> out = new ArrayList<>();
		for (String part : parts) {
			String p = part.strip();
			if (p.isEmpty())
				throw new IllegalArgumentException("empty entry in LoRA targets: " + spec);
			LoraProjection proj = fromKey(p);
			if (!seen.add(proj))
				throw new IllegalArgumentException("duplicate LoRA target: " + proj.key());
			out.add(proj);
		}
		if (out.isEmpty())
			throw new IllegalArgumentException("LoRA targets must not be empty");
		return List.copyOf(out);
	}

	/** Stable deterministic order used for initialization (layer-major, enum order). */
	public static List<LoraProjection> sortedUnique(Iterable<LoraProjection> projections) {
		Set<LoraProjection> set = new LinkedHashSet<>();
		for (LoraProjection p : projections)
			set.add(p);
		List<LoraProjection> ordered = new ArrayList<>();
		for (LoraProjection p : values()) {
			if (set.contains(p))
				ordered.add(p);
		}
		return List.copyOf(ordered);
	}
}
