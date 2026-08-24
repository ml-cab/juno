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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import cab.ml.juno.lora.DoraProjection;
import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterConfig;
import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraMode;

/**
 * Creates DoRA (or plain LoRA/rsLoRA) adapter sets with base-tensor fingerprints
 * and magnitude initialisation from Juno F32 GGUF dequantisation.
 */
public final class DoraInitializer {

	private DoraInitializer() {
	}

	public static LoraAdapterSet create(GgufReader reader, LlamaConfig cfg, Collection<LoraProjection> targets,
			LoraAdapterConfig config, Random rng) throws IOException {
		List<LoraProjection> ordered = LoraProjection.sortedUnique(targets);
		if (ordered.isEmpty())
			throw new IllegalArgumentException("targets must not be empty");

		LoraAdapterSet set = new LoraAdapterSet();
		for (int li = 0; li < cfg.numLayers(); li++) {
			for (LoraProjection proj : ordered) {
				LoraAdapter adapter = new LoraAdapter(config, proj.inDim(cfg), proj.outDim(cfg), rng);
				set.add(li, proj.key(), adapter);
				if (config.mode() == LoraMode.DORA)
					attachDoraState(reader, set, li, proj, adapter.outDim, adapter.inDim);
			}
		}
		return set;
	}

	/**
	 * Attach magnitudes and fingerprints for every DoRA adapter already in
	 * {@code set}. No-ops for plain LoRA entries.
	 */
	public static void attachMissingDoraState(GgufReader reader, LlamaConfig cfg, LoraAdapterSet set)
			throws IOException {
		for (var entry : set.asMap().entrySet()) {
			LoraAdapter a = entry.getValue();
			if (a.mode != LoraMode.DORA)
				continue;
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			LoraProjection proj = LoraProjection.fromKey(LoraAdapterSet.keyProj(key));
			if (set.getMagnitude(layer, proj.key()) == null)
				attachDoraState(reader, set, layer, proj, a.outDim, a.inDim);
		}
	}

	/**
	 * Verify fingerprints for DoRA entries; fail on mismatch.
	 *
	 * @throws IllegalArgumentException if a fingerprint is missing or disagrees
	 */
	public static void verifyFingerprints(GgufReader reader, LoraAdapterSet set) throws IOException {
		for (var entry : set.asMap().entrySet()) {
			if (entry.getValue().mode != LoraMode.DORA)
				continue;
			String key = entry.getKey();
			int layer = LoraAdapterSet.keyLayer(key);
			String projKey = LoraAdapterSet.keyProj(key);
			LoraProjection proj = LoraProjection.fromKey(projKey);
			LoraAdapterSet.BaseTensorFingerprint expected = set.getFingerprint(layer, projKey);
			if (expected == null)
				throw new IllegalArgumentException("DoRA adapter missing base fingerprint: " + key);
			LoraAdapterSet.BaseTensorFingerprint actual = fingerprint(reader, proj.ggufTensorName(layer));
			if (!expected.equals(actual))
				throw new IllegalArgumentException("DoRA base fingerprint mismatch for " + key
						+ " (adapter was trained on a different base tensor)");
		}
	}

	private static void attachDoraState(GgufReader reader, LoraAdapterSet set, int layer, LoraProjection proj,
			int outDim, int inDim) throws IOException {
		String name = proj.ggufTensorName(layer);
		float[] w = reader.tensor(name);
		if (w.length != outDim * inDim)
			throw new IllegalArgumentException("tensor " + name + " shape mismatch for DoRA");
		set.putMagnitude(layer, proj.key(), DoraProjection.magnitudeFromBaseRows(w, outDim, inDim));
		set.putFingerprint(layer, proj.key(), fingerprint(reader, name));
	}

	static LoraAdapterSet.BaseTensorFingerprint fingerprint(GgufReader reader, String tensorName) throws IOException {
		int type = reader.tensorType(tensorName);
		long[] longDims = reader.tensorDims(tensorName);
		int[] dims = new int[longDims.length];
		for (int i = 0; i < longDims.length; i++) {
			if (longDims[i] > Integer.MAX_VALUE)
				throw new IOException("tensor dim too large: " + tensorName);
			dims[i] = (int) longDims[i];
		}
		byte[] raw = reader.tensorRaw(tensorName).data();
		return new LoraAdapterSet.BaseTensorFingerprint(type, dims, sha256(raw));
	}

	private static byte[] sha256(byte[] data) {
		try {
			return MessageDigest.getInstance("SHA-256").digest(data);
		} catch (NoSuchAlgorithmException e) {
			throw new IllegalStateException("SHA-256 not available", e);
		}
	}
}
