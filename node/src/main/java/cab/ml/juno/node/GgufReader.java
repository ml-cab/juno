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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Reads GGUF v2/v3 files and exposes tensor data as float[] arrays.
 *
 * Supported quantisation types: F32 — lossless passthrough F16 — half-precision
 * → float32 (IEEE 754 bit manipulation, no JNI) BF16 — bfloat16 → float32 Q8_0
 * — 8-bit symmetric, block size 32 Q4_0 — 4-bit symmetric, block size 32 Q4_K —
 * 4-bit with per-superblock scale/min, block size 256 (Q4_K_M uses this) Q6_K —
 * 6-bit with per-superblock scale, block size 256
 *
 * Thread-safe after construction — tensor data is loaded on demand then cached.
 *
 * Usage: GgufReader r =
 * GgufReader.open(Path.of("/models/TinyLlama.Q4_K_M.gguf")); LlamaConfig cfg =
 * LlamaConfig.from(r); float[] w = r.tensor("blk.0.attn_q.weight");
 */
public final class GgufReader implements AutoCloseable {

	private static final Logger log = Logger.getLogger(GgufReader.class.getName());

	private static final int GGUF_MAGIC = 0x46554747; // "GGUF"
	private static final int ALIGNMENT = 32;

	// ── GGML quantisation type IDs ───────────────────────────────────────────
	private static final int GGML_TYPE_F32 = 0;
	private static final int GGML_TYPE_F16 = 1;
	private static final int GGML_TYPE_Q4_0 = 2;
	private static final int GGML_TYPE_Q2_K = 10;
	private static final int GGML_TYPE_Q3_K = 11;
	private static final int GGML_TYPE_Q8_0 = 8;
	private static final int GGML_TYPE_Q4_K = 12;
	private static final int GGML_TYPE_Q6_K = 14;
	private static final int GGML_TYPE_Q5_K = 13;
	private static final int GGML_TYPE_BF16 = 30;

	// Metadata value type IDs
	private static final int GGUF_METADATA_VALUE_TYPE_UINT8 = 0;
	private static final int GGUF_METADATA_VALUE_TYPE_INT8 = 1;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT16 = 2;
	private static final int GGUF_METADATA_VALUE_TYPE_INT16 = 3;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT32 = 4;
	private static final int GGUF_METADATA_VALUE_TYPE_INT32 = 5;
	private static final int GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6;
	private static final int GGUF_METADATA_VALUE_TYPE_BOOL = 7;
	private static final int GGUF_METADATA_VALUE_TYPE_STRING = 8;
	private static final int GGUF_METADATA_VALUE_TYPE_ARRAY = 9;
	private static final int GGUF_METADATA_VALUE_TYPE_UINT64 = 10;
	private static final int GGUF_METADATA_VALUE_TYPE_INT64 = 11;
	private static final int GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12;

	private final FileChannel channel;
	private final Map<String, Object> metadata = new HashMap<>();
	private final Map<String, TensorInfo> tensors = new HashMap<>();
	private final long dataOffset; // byte offset where tensor data starts
	private final Map<String, float[]> cache = new HashMap<>();

	// ── Constructor / factory ─────────────────────────────────────────────────

	private GgufReader(FileChannel channel, Map<String, Object> metadata, Map<String, TensorInfo> tensors,
			long dataOffset) {
		this.channel = channel;
		this.metadata.putAll(metadata);
		this.tensors.putAll(tensors);
		this.dataOffset = dataOffset;
	}

	public static GgufReader open(Path file) throws IOException {
		FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);

		// Detect GGUF start offset — plain .gguf files start at 0; .llamafile
		// files are ZIP polyglots with the GGUF stored as an uncompressed entry.
		long ggufOffset = 0L;
		ByteBuffer magic4 = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
		channel.read(magic4, 0);
		magic4.flip();
		int firstMagic = magic4.getInt();
		if (firstMagic != GGUF_MAGIC) {
			log.info("File does not start with GGUF magic (0x" + Integer.toHexString(firstMagic)
					+ ") — trying ZIP/llamafile scan…");
			ggufOffset = findGgufOffsetInZip(channel);
			log.info("Found GGUF data at byte offset " + ggufOffset + " inside llamafile");
		}

		ByteBuffer header = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
		channel.read(header, ggufOffset);
		header.flip();

		int magic = header.getInt();
		int version = header.getInt();
		long tensorCount = header.getLong();
		long kvCount = header.getLong();

		if (magic != GGUF_MAGIC)
			throw new IOException(
					"Not a GGUF file (magic=0x" + Integer.toHexString(magic) + " at offset " + ggufOffset + ")");
		if (version < 2 || version > 3)
			throw new IOException("Unsupported GGUF version: " + version);

		log.info("GGUF v" + version + " — tensors=" + tensorCount + " metadata=" + kvCount);

		// Parse using a streaming position tracker (absolute file positions)
		long[] pos = { ggufOffset + 24L };

		// Read metadata
		Map<String, Object> metadata = new HashMap<>();
		for (long i = 0; i < kvCount; i++) {
			String key = readString(channel, pos);
			int vtype = readInt32(channel, pos);
			Object value = readMetadataValue(channel, pos, vtype);
			metadata.put(key, value);
		}

		// Read tensor info
		Map<String, TensorInfo> tensors = new HashMap<>();
		for (long i = 0; i < tensorCount; i++) {
			String name = readString(channel, pos);
			int ndims = readInt32(channel, pos);
			long[] dims = new long[ndims];
			for (int d = 0; d < ndims; d++)
				dims[d] = readUInt64(channel, pos);
			int type = readInt32(channel, pos);
			long offset = readUInt64(channel, pos);
			long nelems = 1;
			for (long d : dims)
				nelems *= d;
			tensors.put(name, new TensorInfo(name, dims, type, offset, nelems));
		}

		// Align to ALIGNMENT bytes — the GGUF spec aligns relative to the start
		// of the GGUF header, not the start of the file. When the GGUF is
		// embedded inside a llamafile the header starts at ggufOffset, so we
		// must compute the aligned position relative to ggufOffset and then
		// add it back to get the absolute file position.
		long relativePos = pos[0] - ggufOffset;
		long alignedRelative = ((relativePos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
		long aligned = ggufOffset + alignedRelative;

		log.info("Data section starts at byte " + aligned);
		return new GgufReader(channel, metadata, tensors, aligned);
	}

	// ── Public API ────────────────────────────────────────────────────────────

	/** Get a metadata value (String, Number, Boolean, Object[], …). */
	public Object meta(String key) {
		return metadata.get(key);
	}

	public String metaString(String key) {
		Object v = metadata.get(key);
		return v instanceof String s ? s : null;
	}

	public long metaLong(String key, long def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.longValue() : def;
	}

	public int metaInt(String key, int def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.intValue() : def;
	}

	public float metaFloat(String key, float def) {
		Object v = metadata.get(key);
		return v instanceof Number n ? n.floatValue() : def;
	}

	public boolean hasTensor(String name) {
		return tensors.containsKey(name);
	}

	public Map<String, Object> allMetadata() {
		return java.util.Collections.unmodifiableMap(metadata);
	}

	// ── QuantizedTensor ───────────────────────────────────────────────────────

	/**
	 * A tensor held in its original quantized encoding (e.g. Q4_K bytes).
	 *
	 * <p>
	 * Use this instead of {@link #tensor(String)} when you want to keep the weights
	 * compressed and dequantize on-the-fly during matmul, avoiding the enormous
	 * float32 allocation that causes OOM for large models.
	 *
	 * <p>
	 * Memory comparison for one phi-3.5-mini projection matrix (3072 × 3072):
	 * <ul>
	 * <li>{@code tensor()} → float[] = 3072 × 3072 × 4 B ≈ 37.7 MB per layer
	 * <li>{@code tensorRaw()} → Q4_K = 3072 × 12 blocks × 144 B ≈ 5.3 MB per layer
	 * </ul>
	 *
	 * @param name   tensor name as it appears in the GGUF file
	 * @param type   GGML quantisation type ID (0=F32, 1=F16, 8=Q8_0, 12=Q4_K, …)
	 * @param nelems total number of logical scalar elements in the tensor
	 * @param data   raw quantised bytes (NOT dequantised)
	 */
	public record QuantizedTensor(String name, int type, long nelems, byte[] data) {
	}

	/**
	 * Load the raw (quantised) bytes for a tensor without dequantising.
	 *
	 * @throws IllegalArgumentException      if the tensor does not exist
	 * @throws UnsupportedOperationException if the type has no known byte-size
	 *                                       formula
	 */
	public QuantizedTensor tensorRaw(String name) throws IOException {
		TensorInfo info = tensors.get(name);
		if (info == null)
			throw new IllegalArgumentException(
					"Tensor not found: " + name + "  (available: " + tensors.size() + " tensors)");
		long byteCount = rawByteCount(info.type, info.nelems);
		ByteBuffer buf = readBytes(info.offset, byteCount);
		byte[] raw = new byte[(int) byteCount];
		buf.get(raw);
		return new QuantizedTensor(name, info.type, info.nelems, raw);
	}

	/**
	 * Byte size of a quantised tensor in its encoded form.
	 *
	 * @param type   GGML quantisation type ID
	 * @param nelems number of logical scalar elements
	 */
	public static long rawByteCount(int type, long nelems) {
		return switch (type) {
		case GGML_TYPE_F32 -> nelems * 4L;
		case GGML_TYPE_F16, GGML_TYPE_BF16 -> nelems * 2L;
		case GGML_TYPE_Q8_0 -> (nelems / 32L) * 34L;
		case GGML_TYPE_Q4_0 -> (nelems / 32L) * 18L;
		case GGML_TYPE_Q2_K -> (nelems / 256L) * 84L;
		case GGML_TYPE_Q3_K -> (nelems / 256L) * 110L;
		case GGML_TYPE_Q4_K -> (nelems / 256L) * 144L;
		case GGML_TYPE_Q5_K -> (nelems / 256L) * 176L;
		case GGML_TYPE_Q6_K -> (nelems / 256L) * 210L;
		default -> throw new UnsupportedOperationException("No byte-size formula for GGML type " + type);
		};
	}

	/**
	 * Load and dequantize a tensor to float[]. Results are cached — subsequent
	 * calls with the same name are free.
	 */
	public float[] tensor(String name) throws IOException {
		float[] cached = cache.get(name);
		if (cached != null)
			return cached;

		TensorInfo info = tensors.get(name);
		if (info == null)
			throw new IllegalArgumentException(
					"Tensor not found: " + name + "  (available: " + tensors.size() + " tensors)");

		float[] data = loadTensor(info);
		cache.put(name, data);
		return data;
	}

	@Override
	public void close() throws IOException {
		channel.close();
	}

	// ── Tensor loading + dequantisation ───────────────────────────────────────

	private float[] loadTensor(TensorInfo info) throws IOException {
		return switch (info.type) {
		case GGML_TYPE_F32 -> loadF32(info);
		case GGML_TYPE_F16 -> loadF16(info);
		case GGML_TYPE_BF16 -> loadBF16(info);
		case GGML_TYPE_Q8_0 -> loadQ8_0(info);
		case GGML_TYPE_Q4_0 -> loadQ4_0(info);
		case GGML_TYPE_Q2_K -> loadQ2_K(info);
		case GGML_TYPE_Q3_K -> loadQ3_K(info);
		case GGML_TYPE_Q4_K -> loadQ4_K(info);
		case GGML_TYPE_Q5_K -> loadQ5_K(info);
		case GGML_TYPE_Q6_K -> loadQ6_K(info);
		default -> throw new UnsupportedOperationException(
				"Unsupported tensor type " + info.type + " for tensor " + info.name);
		};
	}

	private float[] loadF32(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 4);
		float[] out = new float[n];
		buf.asFloatBuffer().get(out);
		return out;
	}

	private float[] loadF16(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 2);
		float[] out = new float[n];
		for (int i = 0; i < n; i++)
			out[i] = f16ToF32(buf.getShort());
		return out;
	}

	private float[] loadBF16(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		ByteBuffer buf = readBytes(info.offset, (long) n * 2);
		float[] out = new float[n];
		for (int i = 0; i < n; i++) {
			int bits = (buf.getShort() & 0xFFFF) << 16;
			out[i] = Float.intBitsToFloat(bits);
		}
		return out;
	}

	// Q8_0: blocks of 32 elements, 2-byte f16 scale + 32 signed bytes
	private float[] loadQ8_0(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int blockSize = 32;
		int blockBytes = 2 + blockSize; // scale f16 + 32 x int8
		int nBlocks = n / blockSize;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			float scale = f16ToF32(buf.getShort());
			for (int i = 0; i < blockSize; i++)
				out[oi++] = scale * buf.get(); // signed byte
		}
		return out;
	}

	// Q4_0: blocks of 32 elements, 2-byte f16 scale + 16 packed nibbles
	private float[] loadQ4_0(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int blockSize = 32;
		int blockBytes = 2 + blockSize / 2;
		int nBlocks = n / blockSize;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		for (int b = 0; b < nBlocks; b++) {
			float scale = f16ToF32(buf.getShort());
			// Low nibble = first 16, high nibble = second 16
			byte[] qs = new byte[16];
			buf.get(qs);
			for (int i = 0; i < 16; i++)
				out[oi++] = scale * ((qs[i] & 0xF) - 8);
			for (int i = 0; i < 16; i++)
				out[oi++] = scale * ((qs[i] >> 4 & 0xF) - 8);
		}
		return out;
	}

	// Q2_K: superblocks of 256 elements
	// [scales:16 bytes][qs:64 bytes][d:f16][dmin:f16] = 84 bytes per 256 elements
	//
	// Each superblock is split into two halves of 128 elements.
	// Within each half, 8 scale bytes each encode a 4-bit scale (lower nibble)
	// and a 4-bit min (upper nibble). The 32 qs bytes hold four 2-bit quant
	// values per byte. Dequant formula (mirrors llama.cpp dequantize_row_q2_K):
	//   out[l+ 0] = d*(sc[0]&0xF)*((q[l   ]>>0)&3) - dmin*(sc[0]>>4)
	//   out[l+16] = d*(sc[1]&0xF)*((q[l   ]>>2)&3) - dmin*(sc[1]>>4)
	//   out[l+32] = d*(sc[2]&0xF)*((q[l+16]>>0)&3) - dmin*(sc[2]>>4)
	//   out[l+48] = d*(sc[3]&0xF)*((q[l+16]>>2)&3) - dmin*(sc[3]>>4)
	//   out[l+64] = d*(sc[4]&0xF)*((q[l   ]>>4)&3) - dmin*(sc[4]>>4)
	//   out[l+80] = d*(sc[5]&0xF)*((q[l   ]>>6)&3) - dmin*(sc[5]>>4)
	//   out[l+96] = d*(sc[6]&0xF)*((q[l+16]>>4)&3) - dmin*(sc[6]>>4)
	//   out[l+112]= d*(sc[7]&0xF)*((q[l+16]>>6)&3) - dmin*(sc[7]>>4)
	//   for l in 0..15
	private float[] loadQ2_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 84; // 16 + 64 + 2 + 2
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] sc = new byte[16];
		byte[] qs = new byte[64];

		for (int b = 0; b < nBlocks; b++) {
			buf.get(sc);
			buf.get(qs);
			float d    = f16ToF32(buf.getShort());
			float dmin = f16ToF32(buf.getShort());

			// Two halves of 128 elements each
			for (int half = 0; half < 2; half++) {
				int scBase = half * 8;
				int qBase  = half * 32;

				for (int l = 0; l < 16; l++) {
					out[oi + l +   0] = d * (sc[scBase + 0] & 0xF) * ((qs[qBase + l     ]     ) & 3) - dmin * ((sc[scBase + 0] & 0xFF) >> 4);
					out[oi + l +  16] = d * (sc[scBase + 1] & 0xF) * ((qs[qBase + l + 16]     ) & 3) - dmin * ((sc[scBase + 1] & 0xFF) >> 4);
					out[oi + l +  32] = d * (sc[scBase + 2] & 0xF) * ((qs[qBase + l     ] >> 2) & 3) - dmin * ((sc[scBase + 2] & 0xFF) >> 4);
					out[oi + l +  48] = d * (sc[scBase + 3] & 0xF) * ((qs[qBase + l + 16] >> 2) & 3) - dmin * ((sc[scBase + 3] & 0xFF) >> 4);
					out[oi + l +  64] = d * (sc[scBase + 4] & 0xF) * ((qs[qBase + l     ] >> 4) & 3) - dmin * ((sc[scBase + 4] & 0xFF) >> 4);
					out[oi + l +  80] = d * (sc[scBase + 5] & 0xF) * ((qs[qBase + l + 16] >> 4) & 3) - dmin * ((sc[scBase + 5] & 0xFF) >> 4);
					out[oi + l +  96] = d * (sc[scBase + 6] & 0xF) * ((qs[qBase + l     ] >> 6) & 3) - dmin * ((sc[scBase + 6] & 0xFF) >> 4);
					out[oi + l + 112] = d * (sc[scBase + 7] & 0xF) * ((qs[qBase + l + 16] >> 6) & 3) - dmin * ((sc[scBase + 7] & 0xFF) >> 4);
				}
				oi += 128;
			}
		}
		return out;
	}

	// Q3_K: superblocks of 256 elements
	// [hmask:32 bytes][qs:64 bytes][scales:12 bytes][d:f16] = 110 bytes per 256 elements
	//
	// Each element uses 3 bits: 2 low bits from qs, 1 high bit from hmask.
	// 16 groups of 16 elements, each with a signed 6-bit scale (stored biased +32).
	//
	// Loop mirrors llama.cpp dequantize_row_q3_K exactly:
	//   two 128-element halves (qBase=0, qBase=32 in qs);
	//   within each half, 4 bit-shift iterations (shift=0,2,4,6);
	//   hmask bitmask m advances 1→2→4→8→16→32→64→128 across both halves.
	//   signed quant = (low2 | (hmask_bit<<2)) - 4   →  -4..3
	//   output = d * sc[is] * signed_quant
	private float[] loadQ3_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 110; // 32 + 64 + 12 + 2
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] hmask = new byte[32];
		byte[] qs    = new byte[64];
		byte[] scRaw = new byte[12];
		int[]  sc    = new int[16]; // signed scale values (biased -32 already applied)

		for (int b = 0; b < nBlocks; b++) {
			buf.get(hmask);
			buf.get(qs);
			buf.get(scRaw);
			float d = f16ToF32(buf.getShort());

			// Unpack 16 × 6-bit scales from 12 bytes — mirrors llama.cpp kmask1/kmask2 logic
			int aux0 = (scRaw[ 0]&0xFF)|((scRaw[ 1]&0xFF)<<8)|((scRaw[ 2]&0xFF)<<16)|((scRaw[ 3]&0xFF)<<24);
			int aux1 = (scRaw[ 4]&0xFF)|((scRaw[ 5]&0xFF)<<8)|((scRaw[ 6]&0xFF)<<16)|((scRaw[ 7]&0xFF)<<24);
			int aux2 = (scRaw[ 8]&0xFF)|((scRaw[ 9]&0xFF)<<8)|((scRaw[10]&0xFF)<<16)|((scRaw[11]&0xFF)<<24);
			int[] utmp = {
				( aux0        & 0x0f0f0f0f) | (((aux2      ) & 0x03030303) << 4),
				( aux1        & 0x0f0f0f0f) | (((aux2 >>> 2) & 0x03030303) << 4),
				((aux0 >>> 4) & 0x0f0f0f0f) | (((aux2 >>> 4) & 0x03030303) << 4),
				((aux1 >>> 4) & 0x0f0f0f0f) | (((aux2 >>> 6) & 0x03030303) << 4),
			};
			for (int k = 0; k < 4; k++) {
				sc[k*4+0] = ( utmp[k]        & 0xFF) - 32;
				sc[k*4+1] = ((utmp[k] >>>  8) & 0xFF) - 32;
				sc[k*4+2] = ((utmp[k] >>> 16) & 0xFF) - 32;
				sc[k*4+3] = ((utmp[k] >>> 24) & 0xFF) - 32;
			}

			// m is the bitmask into hmask bytes; advances 1,2,4,8,16,32,64,128 across both halves
			int is = 0;
			int m  = 1;
			for (int half = 0; half < 2; half++) {
				int qBase = half * 32;
				int shift = 0;
				for (int j = 0; j < 4; j++) {
					float dl0 = d * sc[is++];
					for (int l = 0; l < 16; l++) {
						int low2 = (qs[qBase + l]      >>> shift) & 3;
						int hi   = ((hmask[l]      & 0xFF) & m) != 0 ? 4 : 0;
						out[oi++] = dl0 * (float) ((low2 | hi) - 4);
					}
					float dl1 = d * sc[is++];
					for (int l = 0; l < 16; l++) {
						int low2 = (qs[qBase + l + 16] >>> shift) & 3;
						int hi   = ((hmask[l + 16] & 0xFF) & m) != 0 ? 4 : 0;
						out[oi++] = dl1 * (float) ((low2 | hi) - 4);
					}
					shift += 2;
					m <<= 1;
				}
			}
		}
		return out;
	}

	// Q4_K: superblocks of 256 elements
	// [d:f16][dmin:f16][scales:12 bytes][qs:128 bytes] = 144 bytes per 256 elements
	private float[] loadQ4_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 144;
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] qs = new byte[128];
		byte[] sc = new byte[12];

		for (int b = 0; b < nBlocks; b++) {
			float d = f16ToF32(buf.getShort());
			float dmin = f16ToF32(buf.getShort());
			buf.get(sc);
			buf.get(qs);

			// 8 sub-blocks of 32 elements each, grouped as 4 pairs of 64
			int qi = 0;
			for (int g = 0; g < QK_K; g += 64) {
				// pair index within the 8 sub-blocks (g/32 and g/32+1)
				int s0 = g / 32;
				int s1 = s0 + 1;
				float scale0 = d * getScale4K(sc, s0);
				float min0 = dmin * getMin4K(sc, s0);
				float scale1 = d * getScale4K(sc, s1);
				float min1 = dmin * getMin4K(sc, s1);

				// First 32: low nibbles of qs[qi..qi+32)
				for (int i = 0; i < 32; i++)
					out[oi++] = scale0 * (qs[qi + i] & 0x0F) - min0;
				// Second 32: high nibbles of qs[qi..qi+32)
				for (int i = 0; i < 32; i++)
					out[oi++] = scale1 * ((qs[qi + i] >> 4) & 0x0F) - min1;
				qi += 32;
			}
		}
		return out;
	}

	// Q5_K: superblocks of 256 elements
	// [d:f16][dmin:f16][scales:12 bytes][qh:32 bytes][qs:128 bytes] = 176 bytes per
	// 256 elements
	//
	// Layout mirrors Q4_K's grouped nibble scheme (not interleaved):
	// 4 groups of 64 elements, each group split into two sub-blocks of 32.
	// For group g (0..3), the 32 qs bytes at offset g*32 serve BOTH sub-blocks:
	// sub-block 2g+0 (first 32): low nibbles of qs[g*32 .. g*32+32)
	// sub-block 2g+1 (second 32): high nibbles of the same qs bytes
	// The qh byte array is 32 bytes; each byte provides one bit per element l
	// (0..31):
	// sub-block 2g+0 uses bit (2*g) of qh[l]
	// sub-block 2g+1 uses bit (2*g+1) of qh[l]
	// This matches llama.cpp: m1=1,m2=2 shifted left by 2 each group iteration.
	private float[] loadQ5_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 176; // 2+2+12+32+128
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] sc = new byte[12];
		byte[] qh = new byte[32];
		byte[] qs = new byte[128];

		for (int b = 0; b < nBlocks; b++) {
			float d = f16ToF32(buf.getShort());
			float dmin = f16ToF32(buf.getShort());
			buf.get(sc);
			buf.get(qh);
			buf.get(qs);

			int qi = 0; // index into qs, advances by 32 per group
			for (int g = 0; g < 4; g++) {
				int s0 = g * 2;
				int s1 = s0 + 1;
				float scale0 = d * getScale4K(sc, s0);
				float min0 = dmin * getMin4K(sc, s0);
				float scale1 = d * getScale4K(sc, s1);
				float min1 = dmin * getMin4K(sc, s1);
				int hiBit0 = g * 2; // bit within qh[l] for first 32
				int hiBit1 = g * 2 + 1; // bit within qh[l] for second 32

				// First 32 of group: low nibbles, qh bit hiBit0
				for (int l = 0; l < 32; l++) {
					int lo = qs[qi + l] & 0x0F;
					int hi = (qh[l] >>> hiBit0) & 1;
					out[oi++] = scale0 * (lo | (hi << 4)) - min0;
				}
				// Second 32 of group: high nibbles of same qs bytes, qh bit hiBit1
				for (int l = 0; l < 32; l++) {
					int lo = (qs[qi + l] >>> 4) & 0x0F;
					int hi = (qh[l] >>> hiBit1) & 1;
					out[oi++] = scale1 * (lo | (hi << 4)) - min1;
				}
				qi += 32;
			}
		}
		return out;
	}

	/**
	 * (12 bytes, 8 scales + 8 mins each 6-bit).
	 */
	private static int getScale4K(byte[] sc, int j) {
		if (j < 4)
			return sc[j] & 0x3F;
		return ((sc[j + 4] & 0x0F) | ((sc[j - 4] & 0xC0) >> 2)) & 0x3F;
	}

	/** Extract 6-bit min[j] from Q4_K scales block. */
	private static int getMin4K(byte[] sc, int j) {
		if (j < 4)
			return sc[j + 4] & 0x3F;
		// sc bytes are signed in Java — mask with 0xFF before >> 4 to prevent
		// arithmetic sign-extension corrupting the upper bits of the result.
		return (((sc[j + 4] & 0xFF) >> 4) | ((sc[j] & 0xC0) >> 2)) & 0x3F;
	}

	// Q6_K: superblocks of 256 elements
	// [ql:128 bytes][qh:64 bytes][scales:16 bytes][d:f16] = 210 bytes per 256
	// elements
	private float[] loadQ6_K(TensorInfo info) throws IOException {
		int n = (int) info.nelems;
		int QK_K = 256;
		int blockBytes = 210;
		int nBlocks = n / QK_K;
		ByteBuffer buf = readBytes(info.offset, (long) nBlocks * blockBytes);
		float[] out = new float[n];
		int oi = 0;
		byte[] ql = new byte[128];
		byte[] qh = new byte[64];
		byte[] sc = new byte[16];

		for (int b = 0; b < nBlocks; b++) {
			buf.get(ql);
			buf.get(qh);
			buf.get(sc);
			float d = f16ToF32(buf.getShort());

			// Port of llama.cpp dequantize_row_q6_K.
			// Each 256-element block is split into two halves of 128 elements.
			// Within each half, l iterates 0..31 and produces four outputs:
			// out[l+ 0] ← ql[qlBase+l] low nibble | qh[qhBase+l] bits 1:0 → sub-block
			// sc[scBase + l/16]
			// out[l+ 32] ← ql[qlBase+l+32] low nibble | qh[qhBase+l] bits 3:2 → sc[scBase +
			// l/16 + 2]
			// out[l+ 64] ← ql[qlBase+l] high nibble | qh[qhBase+l] bits 5:4 → sc[scBase +
			// l/16 + 4]
			// out[l+ 96] ← ql[qlBase+l+32] high nibble | qh[qhBase+l] bits 7:6 → sc[scBase
			// + l/16 + 6]
			// All four share the SAME qh byte qh[qhBase+l]; the earlier flat loop
			// used hi=i/4 which is wrong for outputs at l+32, l+64, l+96.
			for (int half = 0; half < 2; half++) {
				int qlBase = half * 64;
				int qhBase = half * 32;
				int scBase = half * 8;
				for (int l = 0; l < 32; l++) {
					int is = l / 16;
					int qlL = ql[qlBase + l] & 0xFF;
					int qlL2 = ql[qlBase + l + 32] & 0xFF;
					int qhL = qh[qhBase + l] & 0xFF;

					int q1 = (qlL & 0x0F) | (((qhL >> 0) & 3) << 4);
					q1 -= 32;
					int q2 = (qlL2 & 0x0F) | (((qhL >> 2) & 3) << 4);
					q2 -= 32;
					int q3 = (qlL >> 4) | (((qhL >> 4) & 3) << 4);
					q3 -= 32;
					int q4 = (qlL2 >> 4) | (((qhL >> 6) & 3) << 4);
					q4 -= 32;

					// sc[] is int8 — Java bytes are signed, which is what we want.
					float d1 = d * sc[scBase + is];
					float d2 = d * sc[scBase + is + 2];
					float d3 = d * sc[scBase + is + 4];
					float d4 = d * sc[scBase + is + 6];

					out[oi + l] = d1 * q1;
					out[oi + l + 32] = d2 * q2;
					out[oi + l + 64] = d3 * q3;
					out[oi + l + 96] = d4 * q4;
				}
				oi += 128;
			}
		}
		return out;
	}

	// ── Llamafile / ZIP polyglot support ─────────────────────────────────────

	/**
	 * Locate the byte offset of a GGUF file stored uncompressed inside a ZIP
	 * archive (the llamafile format).
	 *
	 * Algorithm: 1. Scan the last ≤65557 bytes for the ZIP End-of-Central-Directory
	 * signature (0x06054b50, little-endian). 2. Read the central-directory offset +
	 * size from the EOCD record. 3. Walk central-directory entries looking for one
	 * whose filename ends with ".gguf". 4. Read the matching local-file-header to
	 * determine where the raw (uncompressed) GGUF bytes begin.
	 *
	 * Only ZIP32 is required here — TinyLlama Q5_K_M is ~700 MB which fits
	 * comfortably within ZIP32 limits.
	 */
	static long findGgufOffsetInZip(FileChannel channel) throws IOException {
		long fileSize = channel.size();

		// ── Step 1: find EOCD ────────────────────────────────────────────────
		// EOCD is 22 bytes + optional comment (max 65535 bytes).
		// We scan the last 65557 bytes backwards for the signature 0x06054b50.
		int searchLen = (int) Math.min(fileSize, 65535 + 22);
		long searchStart = fileSize - searchLen;

		ByteBuffer tail = ByteBuffer.allocate(searchLen).order(ByteOrder.LITTLE_ENDIAN);
		while (tail.hasRemaining()) {
			int r = channel.read(tail, searchStart + tail.position());
			if (r < 0)
				break;
		}
		tail.flip();
		int actualLen = tail.limit();

		// Scan backwards for EOCD signature.
		// Accept a candidate if it looks geometrically consistent OR if its
		// cd-offset/size fields carry the ZIP64 sentinel value 0xFFFFFFFF.
		int eocdIdx = -1;
		for (int i = actualLen - 22; i >= 0; i--) {
			if ((tail.getInt(i) & 0xFFFFFFFFL) != 0x06054b50L)
				continue;

			long candidateCdSize = tail.getInt(i + 12) & 0xFFFFFFFFL;
			long candidateCdOffset = tail.getInt(i + 16) & 0xFFFFFFFFL;
			long eocdAbsPos = searchStart + i;

			// ZIP64 sentinel: at least one field is 0xFFFFFFFF — defer geometry
			// check to after we've read the real values from the ZIP64 EOCD.
			boolean zip64 = (candidateCdSize == 0xFFFFFFFFL || candidateCdOffset == 0xFFFFFFFFL);
			if (!zip64) {
				if (candidateCdSize == 0)
					continue;
				if (candidateCdSize > eocdAbsPos)
					continue;
				if (candidateCdOffset + candidateCdSize > eocdAbsPos)
					continue;
				if (candidateCdOffset >= fileSize)
					continue;
			}

			eocdIdx = i;
			break;
		}

		if (eocdIdx < 0) {
			// Real llamafiles (cosmopolitan APE binaries) sometimes have OS-specific
			// PE/Mach-O sections appended AFTER the ZIP's EOCD, pushing it more than
			// 65557 bytes from the end. Fall back to a forward scan: walk the file
			// from the beginning looking for a ZIP local-file-header (0x04034b50)
			// whose filename ends with ".gguf".
			log.info("EOCD backward scan failed — trying forward local-header scan for .gguf entry");
			return findGgufOffsetByForwardScan(channel);
		}

		// ── Step 1b: resolve ZIP64 if needed ─────────────────────────────────
		// ZIP64 EOCD locator sits immediately before the EOCD (20 bytes).
		// ZIP64 EOCD record contains the real 64-bit cdOffset and cdSize.
		//
		// ZIP64 EOCD locator layout:
		// +0 sig 4 = 0x07064b50
		// +4 disk of z64 4
		// +8 z64 EOCD off 8 ← absolute offset of the ZIP64 EOCD record
		// +16 total disks 4
		//
		// ZIP64 EOCD record layout:
		// +0 sig 4 = 0x06064b50
		// +4 record size 8
		// +12 version made 2
		// +14 version needed 2
		// +16 this disk 4
		// +20 CD start disk 4
		// +24 entries here 8
		// +32 entries total 8
		// +40 CD size 8 ← what we want
		// +48 CD offset 8 ← what we want
		long rawCdSize = tail.getInt(eocdIdx + 12) & 0xFFFFFFFFL;
		long rawCdOffset = tail.getInt(eocdIdx + 16) & 0xFFFFFFFFL;
		long cdSize;
		long cdOffset;

		if (rawCdSize == 0xFFFFFFFFL || rawCdOffset == 0xFFFFFFFFL) {
			log.info("ZIP64 sentinels detected — reading ZIP64 EOCD");
			long eocdAbsPos = searchStart + eocdIdx;
			long locatorPos = eocdAbsPos - 20;
			if (locatorPos < 0)
				throw new IOException("ZIP64 EOCD locator would be before start of file");

			ByteBuffer loc = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
			while (loc.hasRemaining()) {
				int r = channel.read(loc, locatorPos + loc.position());
				if (r < 0)
					break;
			}
			loc.flip();

			if (loc.limit() < 20 || (loc.getInt(0) & 0xFFFFFFFFL) != 0x07064b50L)
				throw new IOException("ZIP64 EOCD locator signature not found at offset " + locatorPos);

			long z64EocdAbsPos = loc.getLong(8);
			log.info("ZIP64 EOCD record at abs-offset " + z64EocdAbsPos);

			ByteBuffer z64 = ByteBuffer.allocate(56).order(ByteOrder.LITTLE_ENDIAN);
			while (z64.hasRemaining()) {
				int r = channel.read(z64, z64EocdAbsPos + z64.position());
				if (r < 0)
					break;
			}
			z64.flip();

			if (z64.limit() < 56 || (z64.getInt(0) & 0xFFFFFFFFL) != 0x06064b50L)
				throw new IOException("ZIP64 EOCD record signature not found at offset " + z64EocdAbsPos);

			cdSize = z64.getLong(40);
			cdOffset = z64.getLong(48);
			log.info("ZIP64 EOCD: cdOffset=" + cdOffset + "  cdSize=" + cdSize);
		} else {
			cdSize = rawCdSize;
			cdOffset = rawCdOffset;
			log.info("EOCD found at abs-offset " + (searchStart + eocdIdx) + "  cdOffset=" + cdOffset + "  cdSize="
					+ cdSize);
		}

		if (cdSize == 0)
			throw new IOException("ZIP central directory is empty — no entries in llamafile");
		if (cdOffset >= fileSize)
			throw new IOException("ZIP CD offset " + cdOffset + " is beyond file size " + fileSize);

		// ── Step 2: read central directory ───────────────────────────────────
		ByteBuffer cd = ByteBuffer.allocate((int) cdSize).order(ByteOrder.LITTLE_ENDIAN);
		while (cd.hasRemaining()) {
			int r = channel.read(cd, cdOffset + cd.position());
			if (r < 0)
				break;
		}
		cd.flip();

		// ── Step 3: walk entries looking for *.gguf ──────────────────────────
		// Central-directory entry fixed header is 46 bytes followed by
		// filename / extra / comment variable fields.
		int cdPos = 0;
		while (cdPos + 46 <= cd.limit()) {
			long sig = cd.getInt(cdPos) & 0xFFFFFFFFL;
			if (sig != 0x02014b50L)
				break; // not a CD entry signature — stop

			int fnLen = cd.getShort(cdPos + 28) & 0xFFFF;
			int extraLen = cd.getShort(cdPos + 30) & 0xFFFF;
			int commentLen = cd.getShort(cdPos + 32) & 0xFFFF;
			long localHdrOffset = cd.getInt(cdPos + 42) & 0xFFFFFFFFL;

			int nextEntry = cdPos + 46 + fnLen + extraLen + commentLen;
			if (nextEntry > cd.limit())
				break; // truncated entry — stop

			byte[] fnBytes = new byte[fnLen];
			cd.position(cdPos + 46);
			cd.get(fnBytes);
			String filename = new String(fnBytes, StandardCharsets.UTF_8);
			//2 annoying
			//log.info("ZIP entry: " + filename + "  localHdr=" + localHdrOffset);

			if (filename.endsWith(".gguf")) {
				// ── Step 4: read local file header ───────────────────────────
				// In ZIP64, the local header offset in the CD entry may itself be
				// a sentinel 0xFFFFFFFF — real value is in the CD extra field.
				if (localHdrOffset == 0xFFFFFFFFL) {
					localHdrOffset = readZip64ExtraLocalOffset(cd, cdPos + 46 + fnLen, extraLen);
					log.info("ZIP64 local header offset from extra field: " + localHdrOffset);
				}

				// Local header: 30-byte fixed part + filename + extra.
				ByteBuffer lh = ByteBuffer.allocate(30).order(ByteOrder.LITTLE_ENDIAN);
				while (lh.hasRemaining()) {
					int r = channel.read(lh, localHdrOffset + lh.position());
					if (r < 0)
						break;
				}
				lh.flip();

				if (lh.limit() < 30)
					throw new IOException("Truncated local file header at offset " + localHdrOffset);

				long lhSig = lh.getInt(0) & 0xFFFFFFFFL;
				if (lhSig != 0x04034b50L)
					throw new IOException("Bad local file header signature 0x" + Long.toHexString(lhSig) + " at offset "
							+ localHdrOffset);

				int localFnLen = lh.getShort(26) & 0xFFFF;
				int localExtraLen = lh.getShort(28) & 0xFFFF;

				long dataStart = localHdrOffset + 30L + localFnLen + localExtraLen;
				log.info("GGUF data starts at abs-offset " + dataStart);
				return dataStart;
			}

			cdPos = nextEntry;
		}

		throw new IOException("No .gguf entry found in the ZIP central directory of this llamafile");
	}

	/**
	 * Fallback for llamafiles where the ZIP EOCD is not in the final 65557 bytes
	 * (e.g., cosmopolitan APE binaries with a large PE/Mach-O overlay appended
	 * after the ZIP).
	 *
	 * Algorithm: scan the file forward in 1 MiB chunks, searching for a ZIP local
	 * file header signature (0x04034b50) whose embedded filename ends with ".gguf".
	 * Returns the absolute byte offset of the raw (uncompressed) GGUF data.
	 *
	 * To avoid misidentifying random bytes as a header, we do a lightweight
	 * sanity-check on each candidate: the filename length must be ≤ 512 bytes, the
	 * compression method must be 0 (STORED), and the filename must end with
	 * ".gguf".
	 */
	private static long findGgufOffsetByForwardScan(FileChannel channel) throws IOException {
		final long fileSize = channel.size();
		final int CHUNK = 1 << 20; // 1 MiB
		final long SIG = 0x04034b50L;

		// We keep a 4-byte "carry" so we don't miss a signature that straddles
		// two consecutive chunks.
		byte[] carry = new byte[3];
		int carryLen = 0;
		long chunkStart = 0;

		ByteBuffer buf = ByteBuffer.allocate(CHUNK).order(ByteOrder.LITTLE_ENDIAN);

		while (chunkStart < fileSize) {
			buf.clear();
			// Prepend carry bytes from previous chunk.
			for (int i = 0; i < carryLen; i++)
				buf.put(carry[i]);

			long readPos = chunkStart;
			while (buf.hasRemaining() && readPos < fileSize) {
				int r = channel.read(buf, readPos + (buf.position() - carryLen));
				if (r < 0)
					break;
				readPos += r;
			}
			buf.flip();
			int limit = buf.limit();

			// Search this chunk for the local-file-header signature.
			for (int i = 0; i + 3 < limit; i++) {
				if ((buf.getInt(i) & 0xFFFFFFFFL) != SIG)
					continue;

				// Candidate at absolute offset = chunkStart - carryLen + i.
				long absPos = chunkStart - carryLen + i;

				// Need at least 30 bytes for the fixed local header.
				if (absPos + 30 > fileSize)
					continue;

				// Read the fixed part of the local header (30 bytes).
				ByteBuffer lh = ByteBuffer.allocate(30).order(ByteOrder.LITTLE_ENDIAN);
				while (lh.hasRemaining()) {
					int r = channel.read(lh, absPos + lh.position());
					if (r < 0)
						break;
				}
				lh.flip();
				if (lh.limit() < 30)
					continue;

				// Sanity-check: compression must be STORED (0).
				int compression = lh.getShort(8) & 0xFFFF;
				if (compression != 0)
					continue;

				int fnLen = lh.getShort(26) & 0xFFFF;
				int extraLen = lh.getShort(28) & 0xFFFF;

				// Sanity-check filename length.
				if (fnLen == 0 || fnLen > 512)
					continue;

				// Read the filename.
				if (absPos + 30 + fnLen > fileSize)
					continue;
				ByteBuffer fnBuf = ByteBuffer.allocate(fnLen);
				while (fnBuf.hasRemaining()) {
					int r = channel.read(fnBuf, absPos + 30 + fnBuf.position());
					if (r < 0)
						break;
				}
				fnBuf.flip();
				if (fnBuf.limit() < fnLen)
					continue;

				String filename = new String(fnBuf.array(), 0, fnLen, StandardCharsets.UTF_8);
				log.info("Forward scan — ZIP local entry: " + filename + "  at abs-offset " + absPos);

				if (!filename.endsWith(".gguf"))
					continue;

				long dataStart = absPos + 30L + fnLen + extraLen;
				log.info("Forward scan — GGUF data starts at abs-offset " + dataStart);
				return dataStart;
			}

			// Save the last 3 bytes as carry for the next iteration.
			carryLen = Math.min(3, limit);
			for (int i = 0; i < carryLen; i++)
				carry[i] = buf.get(limit - carryLen + i);

			chunkStart += (limit - carryLen);
		}

		throw new IOException("No valid ZIP EOCD record found — file is neither a GGUF nor a llamafile ZIP");
	}

	/**
	 * Extract the local-header offset from a ZIP64 extra field block.
	 *
	 * ZIP64 extended information extra field (id=0x0001): +0 header id 2 = 0x0001
	 * +2 data size 2 +4 [optional fields in order: uncompressed size(8), compressed
	 * size(8), local header offset(8), disk number(4)]
	 *
	 * Fields are only present if the corresponding CD fixed field held 0xFFFFFFFF.
	 * We scan all extra blocks to find the ZIP64 one, then read the offset field.
	 */
	private static long readZip64ExtraLocalOffset(ByteBuffer cd, int extraStart, int extraLen) throws IOException {
		int pos = extraStart;
		int end = extraStart + extraLen;
		while (pos + 4 <= end) {
			int headerId = cd.getShort(pos) & 0xFFFF;
			int dataSize = cd.getShort(pos + 2) & 0xFFFF;
			if (headerId == 0x0001) {
				// ZIP64 extra block — offset field is at +4+16 (skip two 8-byte
				// size fields that precede it). Guard against truncation.
				int offsetPos = pos + 4 + 16;
				if (offsetPos + 8 <= end)
					return cd.getLong(offsetPos);
			}
			pos += 4 + dataSize;
		}
		throw new IOException("ZIP64 extra field with local-header offset not found");
	}

	// ── I/O helpers ───────────────────────────────────────────────────────────

	private ByteBuffer readBytes(long tensorOffset, long byteCount) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate((int) byteCount).order(ByteOrder.LITTLE_ENDIAN);
		int read = channel.read(buf, dataOffset + tensorOffset);
		if (read != byteCount)
			throw new IOException("Expected " + byteCount + " bytes, got " + read);
		buf.flip();
		return buf;
	}

	private static String readString(FileChannel ch, long[] pos) throws IOException {
		long len = readUInt64(ch, pos);
		ByteBuffer buf = ByteBuffer.allocate((int) len);
		ch.read(buf, pos[0]);
		pos[0] += len;
		return new String(buf.array(), java.nio.charset.StandardCharsets.UTF_8);
	}

	private static int readInt32(FileChannel ch, long[] pos) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(buf, pos[0]);
		pos[0] += 4;
		buf.flip();
		return buf.getInt();
	}

	private static long readUInt64(FileChannel ch, long[] pos) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
		ch.read(buf, pos[0]);
		pos[0] += 8;
		buf.flip();
		return buf.getLong();
	}

	private static Object readMetadataValue(FileChannel ch, long[] pos, int vtype) throws IOException {
		return switch (vtype) {
		case GGUF_METADATA_VALUE_TYPE_UINT8, GGUF_METADATA_VALUE_TYPE_INT8 -> {
			ByteBuffer b = ByteBuffer.allocate(1);
			ch.read(b, pos[0]);
			pos[0]++;
			b.flip();
			yield (int) b.get();
		}
		case GGUF_METADATA_VALUE_TYPE_UINT16, GGUF_METADATA_VALUE_TYPE_INT16 -> {
			ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 2;
			b.flip();
			yield (int) b.getShort();
		}
		case GGUF_METADATA_VALUE_TYPE_UINT32, GGUF_METADATA_VALUE_TYPE_INT32 -> {
			int v = readInt32(ch, pos);
			yield v;
		}
		case GGUF_METADATA_VALUE_TYPE_FLOAT32 -> {
			ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 4;
			b.flip();
			yield b.getFloat();
		}
		case GGUF_METADATA_VALUE_TYPE_BOOL -> {
			ByteBuffer b = ByteBuffer.allocate(1);
			ch.read(b, pos[0]);
			pos[0]++;
			b.flip();
			yield b.get() != 0;
		}
		case GGUF_METADATA_VALUE_TYPE_STRING -> readString(ch, pos);
		case GGUF_METADATA_VALUE_TYPE_UINT64, GGUF_METADATA_VALUE_TYPE_INT64 -> readUInt64(ch, pos);
		case GGUF_METADATA_VALUE_TYPE_FLOAT64 -> {
			ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
			ch.read(b, pos[0]);
			pos[0] += 8;
			b.flip();
			yield b.getDouble();
		}
		case GGUF_METADATA_VALUE_TYPE_ARRAY -> readArray(ch, pos);
		default -> throw new IOException("Unknown metadata type: " + vtype);
		};
	}

	private static Object[] readArray(FileChannel ch, long[] pos) throws IOException {
		int elemType = readInt32(ch, pos);
		long count = readUInt64(ch, pos);
		Object[] arr = new Object[(int) count];
		for (int i = 0; i < count; i++)
			arr[i] = readMetadataValue(ch, pos, elemType);
		return arr;
	}

	// ── F16 → F32 (pure Java, no JNI) ────────────────────────────────────────

	static float f16ToF32(short bits) {
		int s = (bits >> 15) & 1;
		int e = (bits >> 10) & 0x1F;
		int m = bits & 0x3FF;
		int fBits;
		if (e == 0) {
			if (m == 0) {
				fBits = s << 31;
			} else { // subnormal → normalise
				int exp = -14;
				while ((m & 0x400) == 0) {
					m <<= 1;
					exp--;
				}
				m &= 0x3FF;
				fBits = (s << 31) | ((exp + 127) << 23) | (m << 13);
			}
		} else if (e == 31) {
			fBits = (s << 31) | 0x7F800000 | (m << 13); // ±inf or NaN
		} else {
			fBits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
		}
		return Float.intBitsToFloat(fBits);
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	record TensorInfo(String name, long[] dims, int type, long offset, long nelems) {
	}
}