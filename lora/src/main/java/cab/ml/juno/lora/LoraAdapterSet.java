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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Indexed collection of {@link LoraAdapter}s keyed by (layerIndex, projectionName).
 *
 * <p>
 * Use {@code cab.ml.juno.node.LoraQvInitializer#qv} in the node module to build the
 * standard wq/wv adapter set from a loaded {@link cab.ml.juno.node.LlamaConfig}.
 */
public final class LoraAdapterSet {

	private static final int MAGIC = 0x4C4F5241; // "LORA"
	private static final int VERSION = 1;

	private static final char KEY_SEP = ':';

	private final Map<String, LoraAdapter> adapters = new LinkedHashMap<>();

	public void add(int layer, String proj, LoraAdapter adapter) {
		adapters.put(key(layer, proj), adapter);
	}

	public LoraAdapter get(int layer, String proj) {
		return adapters.get(key(layer, proj));
	}

	public List<LoraAdapter> all() {
		return List.copyOf(adapters.values());
	}

	public Map<String, LoraAdapter> asMap() {
		return java.util.Collections.unmodifiableMap(adapters);
	}

	public int size() {
		return adapters.size();
	}

	public void zeroAllGrads() {
		for (LoraAdapter a : adapters.values())
			a.zeroGrad();
	}

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
				out.writeFloat(a.scale * a.rank);
				for (float f : a.a())
					out.writeFloat(f);
				for (float f : a.b())
					out.writeFloat(f);
			}
		}
	}

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

	private static String key(int layer, String proj) {
		return layer + String.valueOf(KEY_SEP) + proj;
	}

	public static int keyLayer(String key) {
		return Integer.parseInt(key.substring(0, key.indexOf(KEY_SEP)));
	}

	public static String keyProj(String key) {
		return key.substring(key.indexOf(KEY_SEP) + 1);
	}
}
