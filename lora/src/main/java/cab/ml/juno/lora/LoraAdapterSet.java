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

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Indexed collection of {@link LoraAdapter}s keyed by (layerIndex, projectionName).
 *
 * <p>
 * Checkpoints: {@link #save(Path)} writes version 2. {@link #load(Path)} reads
 * version 1 and 2. Version 1 loads as standard scaling, legacy-normal init, plain
 * LoRA. Gradients and optimizer state are never serialized.
 *
 * <p>
 * Use {@code cab.ml.juno.node.LoraQvInitializer#qv} in the node module to build the
 * standard wq/wv adapter set from a loaded {@link cab.ml.juno.node.LlamaConfig}.
 */
public final class LoraAdapterSet {

	private static final int MAGIC = 0x4C4F5241; // "LORA"
	private static final int VERSION_V1 = 1;
	private static final int VERSION_V2 = 2;

	private static final char KEY_SEP = ':';

	private final Map<String, LoraAdapter> adapters = new LinkedHashMap<>();
	/** Optional DoRA magnitude vectors keyed identically to adapters. */
	private final Map<String, DoraMagnitude> magnitudes = new LinkedHashMap<>();
	/** Optional adapted-base tensor fingerprints for DoRA entries. */
	private final Map<String, BaseTensorFingerprint> fingerprints = new LinkedHashMap<>();
	/** Bumped when DoRA parameters change so projection caches refresh. */
	private long doraGeneration = 0;

	public void add(int layer, String proj, LoraAdapter adapter) {
		adapters.put(key(layer, proj), adapter);
	}

	public LoraAdapter get(int layer, String proj) {
		return adapters.get(key(layer, proj));
	}

	public void putMagnitude(int layer, String proj, DoraMagnitude magnitude) {
		String k = key(layer, proj);
		LoraAdapter adapter = adapters.get(k);
		if (adapter == null)
			throw new IllegalArgumentException("magnitude requires matching adapter: " + k);
		Objects.requireNonNull(magnitude, "magnitude");
		if (magnitude.length() != adapter.outDim)
			throw new IllegalArgumentException(
					"magnitude length " + magnitude.length() + " != outDim " + adapter.outDim);
		magnitudes.put(k, magnitude);
	}

	public DoraMagnitude getMagnitude(int layer, String proj) {
		return magnitudes.get(key(layer, proj));
	}

	public void putFingerprint(int layer, String proj, BaseTensorFingerprint fingerprint) {
		String k = key(layer, proj);
		if (!adapters.containsKey(k))
			throw new IllegalArgumentException("fingerprint requires matching adapter: " + k);
		fingerprints.put(k, Objects.requireNonNull(fingerprint, "fingerprint"));
	}

	public BaseTensorFingerprint getFingerprint(int layer, String proj) {
		return fingerprints.get(key(layer, proj));
	}

	public List<LoraAdapter> all() {
		return List.copyOf(adapters.values());
	}

	public Map<String, LoraAdapter> asMap() {
		return java.util.Collections.unmodifiableMap(adapters);
	}

	public Map<String, DoraMagnitude> magnitudes() {
		return java.util.Collections.unmodifiableMap(magnitudes);
	}

	public Map<String, BaseTensorFingerprint> fingerprints() {
		return java.util.Collections.unmodifiableMap(fingerprints);
	}

	/** Invalidate DoRA norm caches; call after A/B/magnitude updates. */
	public void invalidateDoraCaches() {
		doraGeneration++;
	}

	public long doraGeneration() {
		return doraGeneration;
	}

	public int size() {
		return adapters.size();
	}

	public void zeroAllGrads() {
		for (LoraAdapter a : adapters.values())
			a.zeroGrad();
		for (DoraMagnitude m : magnitudes.values())
			m.zeroGrad();
	}

	/**
	 * Reinitialize every adapter in this set. Keys present in {@code fresh} copy
	 * those weights; any extra keys (e.g. leftover projections) are reinitialized
	 * locally so ΔW returns to zero. Magnitudes/fingerprints for matching keys are
	 * copied when present in {@code fresh}. Always bumps
	 * {@link #doraGeneration()} so DoRA projection coefficient caches refresh.
	 *
	 * @return number of adapters reset
	 */
	public int resetFrom(LoraAdapterSet fresh, Random rng) {
		int n = 0;
		for (var entry : adapters.entrySet()) {
			String k = entry.getKey();
			LoraAdapter src = fresh.adapters.get(k);
			if (src != null)
				entry.getValue().copyWeightsFrom(src);
			else
				entry.getValue().reinitialize(rng);

			DoraMagnitude srcMag = fresh.magnitudes.get(k);
			if (srcMag != null) {
				DoraMagnitude dst = magnitudes.get(k);
				if (dst != null)
					dst.copyFrom(srcMag);
				else
					magnitudes.put(k, srcMag.copy());
			}
			BaseTensorFingerprint fp = fresh.fingerprints.get(k);
			if (fp != null)
				fingerprints.put(k, fp);
			n++;
		}
		// DoRA keeps row coefficients in DoraProjection until generation bumps;
		// without this, /reset leaves trained magnitude scaling in effect.
		invalidateDoraCaches();
		return n;
	}

	/** Write a version-2 checkpoint (length-delimited entries). */
	public void save(Path path) throws IOException {
		saveV2(path);
	}

	/**
	 * Write a version-1 checkpoint for tooling that cannot read v2. Encodes the
	 * effective scale as a transformed legacy alpha ({@code scale * rank}) so
	 * standard v1 loaders reconstruct the same scale. Rejects DoRA because v1
	 * cannot represent magnitude semantics.
	 */
	public void saveLegacyV1(Path path) throws IOException {
		for (LoraAdapter a : adapters.values()) {
			if (a.mode == LoraMode.DORA)
				throw new IllegalStateException("cannot export DoRA adapters as legacy v1");
		}
		try (var out = new DataOutputStream(Files.newOutputStream(path))) {
			out.writeInt(MAGIC);
			out.writeInt(VERSION_V1);
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

	private void saveV2(Path path) throws IOException {
		try (var out = new DataOutputStream(Files.newOutputStream(path))) {
			out.writeInt(MAGIC);
			out.writeInt(VERSION_V2);
			out.writeInt(adapters.size());
			for (var entry : adapters.entrySet()) {
				byte[] payload = encodeV2Entry(entry.getKey(), entry.getValue());
				out.writeInt(payload.length);
				out.write(payload);
			}
		}
	}

	private byte[] encodeV2Entry(String key, LoraAdapter a) throws IOException {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		try (DataOutputStream out = new DataOutputStream(bos)) {
			byte[] keyBytes = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
			out.writeInt(keyBytes.length);
			out.write(keyBytes);
			out.writeInt(a.rank);
			out.writeInt(a.inDim);
			out.writeInt(a.outDim);
			out.writeFloat(a.alpha);
			out.writeInt(a.scaling.ordinal());
			out.writeInt(a.initialization.ordinal());
			out.writeInt(a.mode.ordinal());
			for (float f : a.a())
				out.writeFloat(f);
			for (float f : a.b())
				out.writeFloat(f);

			DoraMagnitude mag = magnitudes.get(key);
			boolean hasMag = mag != null;
			out.writeBoolean(hasMag);
			if (hasMag) {
				for (float f : mag.values())
					out.writeFloat(f);
			}

			BaseTensorFingerprint fp = fingerprints.get(key);
			boolean hasFp = fp != null;
			out.writeBoolean(hasFp);
			if (hasFp)
				fp.write(out);

			// Reserved length-delimited extension for Tier 5 metadata.
			out.writeInt(0);
		}
		return bos.toByteArray();
	}

	public static LoraAdapterSet load(Path path) throws IOException {
		LoraAdapterSet set = new LoraAdapterSet();
		try (var in = new DataInputStream(Files.newInputStream(path))) {
			int magic = in.readInt();
			if (magic != MAGIC)
				throw new IOException("Not a LoRA checkpoint (magic=0x" + Integer.toHexString(magic) + ")");
			int version = in.readInt();
			if (version == VERSION_V1)
				loadV1(set, in);
			else if (version == VERSION_V2)
				loadV2(set, in);
			else
				throw new IOException("Unsupported LoRA checkpoint version: " + version);

			if (in.read() != -1)
				throw new IOException("Trailing bytes after LoRA checkpoint");
		}
		return set;
	}

	private static void loadV1(LoraAdapterSet set, DataInputStream in) throws IOException {
		int count = in.readInt();
		if (count < 0)
			throw new IOException("Negative adapter count: " + count);
		for (int i = 0; i < count; i++) {
			String key = readKey(in);
			if (set.adapters.containsKey(key))
				throw new IOException("Duplicate adapter key: " + key);
			int rank = in.readInt();
			int inDim = in.readInt();
			int outDim = in.readInt();
			float alpha = in.readFloat();
			validateDims(rank, inDim, outDim, alpha);
			float[] aArr = readFloats(in, Math.multiplyExact(rank, inDim));
			float[] bArr = readFloats(in, Math.multiplyExact(outDim, rank));
			// v1: standard scaling, legacy-normal provenance, plain LoRA
			LoraAdapter adapter = LoraAdapter.fromWeights(LoraAdapterConfig.legacy(rank, alpha), inDim, outDim, aArr,
					bArr);
			set.adapters.put(key, adapter);
		}
	}

	private static void loadV2(LoraAdapterSet set, DataInputStream in) throws IOException {
		int count = in.readInt();
		if (count < 0)
			throw new IOException("Negative adapter count: " + count);
		for (int i = 0; i < count; i++) {
			int entryLen = in.readInt();
			if (entryLen < 0)
				throw new IOException("Negative entry length: " + entryLen);
			byte[] payload = in.readNBytes(entryLen);
			if (payload.length != entryLen)
				throw new EOFException("Truncated LoRA entry: expected " + entryLen + " bytes");
			decodeV2Entry(set, payload);
		}
	}

	private static void decodeV2Entry(LoraAdapterSet set, byte[] payload) throws IOException {
		try (DataInputStream in = new DataInputStream(new java.io.ByteArrayInputStream(payload))) {
			String key = readKey(in);
			if (set.adapters.containsKey(key))
				throw new IOException("Duplicate adapter key: " + key);

			int rank = in.readInt();
			int inDim = in.readInt();
			int outDim = in.readInt();
			float alpha = in.readFloat();
			LoraScaling scaling = LoraScaling.fromId(in.readInt());
			LoraInitialization initialization = LoraInitialization.fromId(in.readInt());
			LoraMode mode = LoraMode.fromId(in.readInt());
			validateDims(rank, inDim, outDim, alpha);

			int aLen = Math.multiplyExact(rank, inDim);
			int bLen = Math.multiplyExact(outDim, rank);
			float[] aArr = readFloats(in, aLen);
			float[] bArr = readFloats(in, bLen);
			requireFinite(aArr, "A");
			requireFinite(bArr, "B");

			LoraAdapterConfig config = LoraAdapterConfig.of(rank, alpha, scaling, initialization, mode);
			LoraAdapter adapter = LoraAdapter.fromWeights(config, inDim, outDim, aArr, bArr);
			set.adapters.put(key, adapter);

			boolean hasMag = in.readBoolean();
			if (hasMag) {
				float[] mag = readFloats(in, outDim);
				requireFinite(mag, "magnitude");
				set.magnitudes.put(key, DoraMagnitude.fromValues(mag));
			} else if (mode == LoraMode.DORA) {
				throw new IOException("DoRA entry missing magnitude: " + key);
			}

			boolean hasFp = in.readBoolean();
			if (hasFp)
				set.fingerprints.put(key, BaseTensorFingerprint.read(in));

			int extLen = in.readInt();
			if (extLen < 0)
				throw new IOException("Negative extension length: " + extLen);
			if (extLen > 0) {
				byte[] ext = in.readNBytes(extLen);
				if (ext.length != extLen)
					throw new EOFException("Truncated extension block");
				// Reserved for Tier 5 — ignore unknown extensions for forward compat.
			}

			if (in.read() != -1)
				throw new IOException("Trailing bytes inside LoRA entry: " + key);
		} catch (IllegalArgumentException e) {
			throw new IOException("Corrupt LoRA entry: " + e.getMessage(), e);
		}
	}

	private static String readKey(DataInputStream in) throws IOException {
		int keyLen = in.readInt();
		if (keyLen < 1 || keyLen > 1_048_576)
			throw new IOException("Invalid key length: " + keyLen);
		byte[] keyBytes = in.readNBytes(keyLen);
		if (keyBytes.length != keyLen)
			throw new EOFException("Truncated key");
		return new String(keyBytes, java.nio.charset.StandardCharsets.UTF_8);
	}

	private static float[] readFloats(DataInputStream in, int n) throws IOException {
		if (n < 0 || n > 100_000_000)
			throw new IOException("Invalid float array length: " + n);
		float[] arr = new float[n];
		for (int j = 0; j < n; j++)
			arr[j] = in.readFloat();
		return arr;
	}

	private static void validateDims(int rank, int inDim, int outDim, float alpha) throws IOException {
		if (rank < 1 || inDim < 1 || outDim < 1)
			throw new IOException("Invalid adapter dimensions rank=" + rank + " in=" + inDim + " out=" + outDim);
		if (!Float.isFinite(alpha))
			throw new IOException("Non-finite alpha: " + alpha);
	}

	private static void requireFinite(float[] arr, String label) throws IOException {
		for (float f : arr) {
			if (!Float.isFinite(f))
				throw new IOException("Non-finite " + label + " value");
		}
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

	/**
	 * SHA-256 fingerprint of an adapted base GGUF tensor (raw bytes + type + dims).
	 */
	public static final class BaseTensorFingerprint {

		private final int tensorType;
		private final int[] dims;
		private final byte[] sha256;

		public BaseTensorFingerprint(int tensorType, int[] dims, byte[] sha256) {
			if (dims == null || dims.length == 0)
				throw new IllegalArgumentException("dims must be non-empty");
			if (sha256 == null || sha256.length != 32)
				throw new IllegalArgumentException("sha256 must be 32 bytes");
			this.tensorType = tensorType;
			this.dims = Arrays.copyOf(dims, dims.length);
			this.sha256 = Arrays.copyOf(sha256, sha256.length);
		}

		public int tensorType() {
			return tensorType;
		}

		public int[] dims() {
			return Arrays.copyOf(dims, dims.length);
		}

		public byte[] sha256() {
			return Arrays.copyOf(sha256, sha256.length);
		}

		void write(DataOutputStream out) throws IOException {
			out.writeInt(tensorType);
			out.writeInt(dims.length);
			for (int d : dims)
				out.writeInt(d);
			out.write(sha256);
		}

		static BaseTensorFingerprint read(DataInputStream in) throws IOException {
			int tensorType = in.readInt();
			int dimCount = in.readInt();
			if (dimCount < 1 || dimCount > 8)
				throw new IOException("Invalid fingerprint dim count: " + dimCount);
			int[] dims = new int[dimCount];
			for (int i = 0; i < dimCount; i++)
				dims[i] = in.readInt();
			byte[] sha = in.readNBytes(32);
			if (sha.length != 32)
				throw new EOFException("Truncated fingerprint hash");
			return new BaseTensorFingerprint(tensorType, dims, sha);
		}

		@Override
		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof BaseTensorFingerprint other))
				return false;
			return tensorType == other.tensorType && Arrays.equals(dims, other.dims)
					&& Arrays.equals(sha256, other.sha256);
		}

		@Override
		public int hashCode() {
			return 31 * (31 * tensorType + Arrays.hashCode(dims)) + Arrays.hashCode(sha256);
		}
	}
}
