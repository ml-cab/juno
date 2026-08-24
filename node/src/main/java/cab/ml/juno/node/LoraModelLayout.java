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
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Architecture-specific mapping from logical LoRA projections to physical GGUF
 * tensors. Checkpoint keys remain logical ({@code wq}, {@code wk}, …); layouts
 * resolve physical names and dimensions at load/merge time.
 */
public final class LoraModelLayout {

	private final String architecture;
	private final int numLayers;
	private final Map<Integer, EnumMap<LoraProjection, LoraProjectionBinding>> byLayer;

	private LoraModelLayout(String architecture, int numLayers,
			Map<Integer, EnumMap<LoraProjection, LoraProjectionBinding>> byLayer) {
		this.architecture = Objects.requireNonNull(architecture);
		this.numLayers = numLayers;
		this.byLayer = byLayer;
	}

	public String architecture() {
		return architecture;
	}

	public int numLayers() {
		return numLayers;
	}

	public LoraProjectionBinding binding(int layer, LoraProjection projection) {
		EnumMap<LoraProjection, LoraProjectionBinding> m = byLayer.get(layer);
		if (m == null)
			throw new IllegalArgumentException("layer " + layer + " out of range [0," + numLayers + ")");
		LoraProjectionBinding b = m.get(projection);
		if (b == null)
			throw new IllegalArgumentException("no binding for " + projection.key() + " at layer " + layer);
		return b;
	}

	public int inDim(int layer, LoraProjection projection) {
		return binding(layer, projection).inDim();
	}

	public int outDim(int layer, LoraProjection projection) {
		return binding(layer, projection).outDim();
	}

	/** All bindings that patch the given physical tensor, in stable projection order. */
	public List<LoraProjectionBinding> bindingsForPhysical(String physicalName) {
		List<LoraProjectionBinding> out = new ArrayList<>();
		for (int li = 0; li < numLayers; li++) {
			EnumMap<LoraProjection, LoraProjectionBinding> m = byLayer.get(li);
			for (LoraProjection p : LoraProjection.values()) {
				LoraProjectionBinding b = m.get(p);
				if (b != null && b.physicalName().equals(physicalName))
					out.add(b);
			}
		}
		return Collections.unmodifiableList(out);
	}

	/** Every binding across all layers, layer-major then enum order. */
	public List<LoraProjectionBinding> allBindings() {
		List<LoraProjectionBinding> out = new ArrayList<>();
		for (int li = 0; li < numLayers; li++) {
			EnumMap<LoraProjection, LoraProjectionBinding> m = byLayer.get(li);
			for (LoraProjection p : LoraProjection.values())
				out.add(m.get(p));
		}
		return Collections.unmodifiableList(out);
	}

	public static LoraModelLayout llama(LlamaConfig cfg) {
		return denseSeparate(cfg.architecture() != null ? cfg.architecture() : "llama", cfg.numLayers(),
				cfg.hiddenDim(), cfg.kvDim(), cfg.hiddenDim(), cfg.intermediateSize());
	}

	public static LoraModelLayout qwen2(LlamaConfig cfg) {
		String arch = cfg.architecture() != null ? cfg.architecture() : "qwen2";
		return denseSeparate(arch, cfg.numLayers(), cfg.hiddenDim(), cfg.kvDim(), cfg.hiddenDim(),
				cfg.intermediateSize());
	}

	/**
	 * Phi-3 fused QKV ({@code attn_qkv.weight}) and fused gate/up
	 * ({@code ffn_up.weight}).
	 */
	public static LoraModelLayout phi3(LlamaConfig cfg) {
		int H = cfg.hiddenDim();
		int KV = cfg.kvDim();
		int I = cfg.intermediateSize();
		int L = cfg.numLayers();
		Map<Integer, EnumMap<LoraProjection, LoraProjectionBinding>> byLayer = new LinkedHashMap<>();
		for (int li = 0; li < L; li++) {
			String qkv = "blk." + li + ".attn_qkv.weight";
			String gateUp = "blk." + li + ".ffn_up.weight";
			EnumMap<LoraProjection, LoraProjectionBinding> m = new EnumMap<>(LoraProjection.class);
			m.put(LoraProjection.WQ, new LoraProjectionBinding(LoraProjection.WQ, qkv, 0, H, H, H));
			m.put(LoraProjection.WK, new LoraProjectionBinding(LoraProjection.WK, qkv, H, KV, H, KV));
			m.put(LoraProjection.WV, new LoraProjectionBinding(LoraProjection.WV, qkv, H + KV, KV, H, KV));
			m.put(LoraProjection.WO,
					new LoraProjectionBinding(LoraProjection.WO, "blk." + li + ".attn_output.weight", 0, H, H, H));
			m.put(LoraProjection.WGATE, new LoraProjectionBinding(LoraProjection.WGATE, gateUp, 0, I, H, I));
			m.put(LoraProjection.WUP, new LoraProjectionBinding(LoraProjection.WUP, gateUp, I, I, H, I));
			m.put(LoraProjection.WDOWN,
					new LoraProjectionBinding(LoraProjection.WDOWN, "blk." + li + ".ffn_down.weight", 0, H, I, H));
			byLayer.put(li, m);
		}
		return new LoraModelLayout("phi3", L, byLayer);
	}

	/** Dense Qwen3: WQ out / WO in use {@link Qwen3Config#qDim()}. */
	public static LoraModelLayout qwen3(Qwen3Config cfg) {
		return denseSeparate(cfg.architecture(), cfg.numLayers(), cfg.hiddenDim(), cfg.kvDim(), cfg.qDim(),
				cfg.intermediateSize());
	}

	/**
	 * Resolve layout from GGUF {@code general.architecture}. Throws for unsupported
	 * training architectures (MoE, gemma, unknown).
	 */
	public static LoraModelLayout forArchitecture(String architecture, LlamaConfig cfg) {
		String a = normalize(architecture);
		return switch (a) {
		case "llama", "mistral", "tinyllama" -> llama(cfg);
		case "qwen2", "qwen2.5" -> qwen2(cfg);
		case "phi3" -> phi3(cfg);
		case "qwen3" -> {
			if (cfg.architecture() == null || !"qwen3".equalsIgnoreCase(cfg.architecture())) {
				// Rebuild with qwen3 headDim if caller only has LlamaConfig — qDim may equal H.
				yield denseSeparate("qwen3", cfg.numLayers(), cfg.hiddenDim(), cfg.kvDim(),
						cfg.numHeads() * cfg.headDim(), cfg.intermediateSize());
			}
			yield denseSeparate("qwen3", cfg.numLayers(), cfg.hiddenDim(), cfg.kvDim(),
					cfg.numHeads() * cfg.headDim(), cfg.intermediateSize());
		}
		default -> throw new IllegalArgumentException(
				"LoRA layout unsupported for architecture '" + a + "'");
		};
	}

	public static LoraModelLayout forArchitecture(String architecture, Qwen3Config cfg) {
		String a = normalize(architecture);
		if (!"qwen3".equals(a))
			throw new IllegalArgumentException("expected qwen3, got " + a);
		return qwen3(cfg);
	}

	private static LoraModelLayout denseSeparate(String arch, int layers, int hidden, int kvDim, int qDim,
			int intermediate) {
		Map<Integer, EnumMap<LoraProjection, LoraProjectionBinding>> byLayer = new LinkedHashMap<>();
		for (int li = 0; li < layers; li++) {
			EnumMap<LoraProjection, LoraProjectionBinding> m = new EnumMap<>(LoraProjection.class);
			m.put(LoraProjection.WQ, new LoraProjectionBinding(LoraProjection.WQ,
					"blk." + li + ".attn_q.weight", 0, qDim, hidden, qDim));
			m.put(LoraProjection.WK, new LoraProjectionBinding(LoraProjection.WK,
					"blk." + li + ".attn_k.weight", 0, kvDim, hidden, kvDim));
			m.put(LoraProjection.WV, new LoraProjectionBinding(LoraProjection.WV,
					"blk." + li + ".attn_v.weight", 0, kvDim, hidden, kvDim));
			m.put(LoraProjection.WO, new LoraProjectionBinding(LoraProjection.WO,
					"blk." + li + ".attn_output.weight", 0, hidden, qDim, hidden));
			m.put(LoraProjection.WGATE, new LoraProjectionBinding(LoraProjection.WGATE,
					"blk." + li + ".ffn_gate.weight", 0, intermediate, hidden, intermediate));
			m.put(LoraProjection.WUP, new LoraProjectionBinding(LoraProjection.WUP,
					"blk." + li + ".ffn_up.weight", 0, intermediate, hidden, intermediate));
			m.put(LoraProjection.WDOWN, new LoraProjectionBinding(LoraProjection.WDOWN,
					"blk." + li + ".ffn_down.weight", 0, hidden, intermediate, hidden));
			byLayer.put(li, m);
		}
		return new LoraModelLayout(arch, layers, byLayer);
	}

	static String normalize(String architecture) {
		if (architecture == null || architecture.isBlank())
			return "llama";
		return architecture.toLowerCase(Locale.ROOT).strip();
	}
}
