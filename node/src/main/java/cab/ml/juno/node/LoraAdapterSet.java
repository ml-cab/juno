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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Indexed collection of {@link LoraAdapter}s keyed by (layerIndex,
 * projectionName).
 *
 * <h3>Naming convention for projectionName</h3>
 * <ul>
 * <li>{@code "wq"} — query projection (most common LoRA target)
 * <li>{@code "wv"} — value projection (second most common)
 * <li>{@code "wk"} — key projection (optional, adds little benefit in practice)
 * <li>{@code "wo"} — attention output projection (optional)
 * </ul>
 *
 * <h3>Building a set</h3>
 * 
 * <pre>
 *   LlamaConfig cfg = ...;
 *   int rank = 8;
 *   float alpha = 8f;
 *   Random rng = new Random(42);
 *
 *   LoraAdapterSet adapters = new LoraAdapterSet();
 *   for (int li = 0; li < cfg.numLayers(); li++) {
 *       adapters.add(li, "wq", new LoraAdapter(rank, cfg.hiddenDim(), cfg.hiddenDim(), alpha, rng));
 *       adapters.add(li, "wv", new LoraAdapter(rank, cfg.hiddenDim(), cfg.kvDim(), alpha, rng));
 *   }
 * </pre>
 *
 * <h3>Serialisation format</h3> Simple big-endian binary:
 * 
 * <pre>
 *   [int] MAGIC = 0x4C4F5241  ("LORA")
 *   [int] version = 1
 *   [int] numAdapters
 *   for each adapter:
 *     [int]   keyLen
 *     [bytes] key (UTF-8: "layerIndex:projName", e.g. "3:wq")
 *     [int]   rank
 *     [int]   inDim
 *     [int]   outDim
 *     [float] alpha     (= scale × rank)
 *     [floats × rank*inDim]  A weights
 *     [floats × outDim*rank] B weights
 * </pre>
 */
public final class LoraAdapterSet {

	private static final int MAGIC = 0x4C4F5241; // "LORA"
	private static final int VERSION = 1;

	/** Separator between layer index and projection name in the map key. */
	private static final char KEY_SEP = ':';

	private final Map<String, LoraAdapter> adapters = new LinkedHashMap<>();

	// ── Mutation ──────────────────────────────────────────────────────────────

	/**
	 * Register an adapter for the given (layer, projection) pair. Replaces any
	 * previously registered adapter for the same key.
	 */
	public void add(int layer, String proj, LoraAdapter adapter) {
		adapters.put(key(layer, proj), adapter);
	}

	// ── Lookup ────────────────────────────────────────────────────────────────

	/**
	 * Returns the adapter for (layer, proj), or {@code null} if none was
	 * registered.
	 */
	public LoraAdapter get(int layer, String proj) {
		return adapters.get(key(layer, proj));
	}

	/** All adapters in insertion order. */
	public List<LoraAdapter> all() {
		return List.copyOf(adapters.values());
	}

	/** Number of registered adapters. */
	public int size() {
		return adapters.size();
	}

	// ── Gradient utilities ────────────────────────────────────────────────────

	/**
	 * Zero every adapter's gradient accumulators. Call before each training step.
	 */
	public void zeroAllGrads() {
		for (LoraAdapter a : adapters.values())
			a.zeroGrad();
	}

	// ── Serialisation ─────────────────────────────────────────────────────────

	/**
	 * Save all adapters to a binary checkpoint file.
	 *
	 * @param path destination file (parent directories must exist)
	 */
	public void save(Path path) throws IOException {
		try (var out = new DataOutputStream(Files.newOutputStream(path))) {
			out.writeInt(MAGIC);
			out.writeInt(VERSION);
			out.writeInt(adapters.size());
			for (var entry : adapters.entrySet()) {
				byte[] keyBytes = entry.getKey().getBytes(java.nio.charset.StandardCharsets.UTF_8);
				out.writeInt(keyBytes.length);
				out.write(keyBytes);

				LoraAdapter a = entry.getValue();
				out.writeInt(a.rank);
				out.writeInt(a.inDim);
				out.writeInt(a.outDim);
				out.writeFloat(a.scale * a.rank); // save as alpha = scale × rank
				for (float f : a.a())
					out.writeFloat(f);
				for (float f : a.b())
					out.writeFloat(f);
			}
		}
	}

	/**
	 * Load adapters from a previously saved checkpoint.
	 *
	 * @param path source file written by {@link #save(Path)}
	 * @return restored adapter set
	 */
	public static LoraAdapterSet load(Path path) throws IOException {
		LoraAdapterSet set = new LoraAdapterSet();
		try (var in = new DataInputStream(Files.newInputStream(path))) {
			int magic = in.readInt();
			if (magic != MAGIC)
				throw new IOException("Not a LoRA checkpoint (magic=0x" + Integer.toHexString(magic) + ")");
			int version = in.readInt();
			if (version != VERSION)
				throw new IOException("Unsupported LoRA checkpoint version: " + version);

			int count = in.readInt();
			for (int i = 0; i < count; i++) {
				int keyLen = in.readInt();
				byte[] keyBytes = in.readNBytes(keyLen);
				String key = new String(keyBytes, java.nio.charset.StandardCharsets.UTF_8);

				int rank = in.readInt();
				int inDim = in.readInt();
				int outDim = in.readInt();
				float alpha = in.readFloat();

				float[] aArr = new float[rank * inDim];
				float[] bArr = new float[outDim * rank];
				for (int j = 0; j < aArr.length; j++)
					aArr[j] = in.readFloat();
				for (int j = 0; j < bArr.length; j++)
					bArr[j] = in.readFloat();

				LoraAdapter adapter = LoraAdapter.fromWeights(rank, inDim, outDim, alpha, aArr, bArr);
				set.adapters.put(key, adapter);
			}
		}
		return set;
	}

	// ── Factory helpers ───────────────────────────────────────────────────────

	/**
	 * Convenience factory: create adapters on wq and wv for every layer in a
	 * LlamaConfig. This is the standard LoRA configuration from the original paper.
	 *
	 * @param cfg   model configuration (supplies hiddenDim, kvDim, numLayers)
	 * @param rank  LoRA rank (4, 8, or 16 are common; higher = more parameters)
	 * @param alpha scaling factor (set equal to rank for scale = 1.0)
	 * @param rng   random source for A initialisation
	 */
	public static LoraAdapterSet qv(LlamaConfig cfg, int rank, float alpha, Random rng) {
		LoraAdapterSet set = new LoraAdapterSet();
		for (int li = 0; li < cfg.numLayers(); li++) {
			set.add(li, "wq", new LoraAdapter(rank, cfg.hiddenDim(), cfg.hiddenDim(), alpha, rng));
			set.add(li, "wv", new LoraAdapter(rank, cfg.hiddenDim(), cfg.kvDim(), alpha, rng));
		}
		return set;
	}

	// ── Internal ──────────────────────────────────────────────────────────────

	private static String key(int layer, String proj) {
		return layer + String.valueOf(KEY_SEP) + proj;
	}

	/** Parse the layer index from a stored key (e.g. "3:wq" → 3). */
	static int keyLayer(String key) {
		return Integer.parseInt(key.substring(0, key.indexOf(KEY_SEP)));
	}

	/** Parse the projection name from a stored key (e.g. "3:wq" → "wq"). */
	static String keyProj(String key) {
		return key.substring(key.indexOf(KEY_SEP) + 1);
	}
}