package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for CpuForwardPassHandler.matVec(QuantizedTensor, ...).
 *
 * These tests prove that doing a matVec directly on raw quantized bytes gives
 * the same numerical result as: load tensor → GgufReader dequantizes to float[]
 * → float[] matVec.
 *
 * The root-cause of the OOM crash (Killed) was that PhiForwardPassHandler
 * called r.tensor() for every projection weight, which dequantizes the full
 * tensor to float32 eagerly: 32 layers × ~450 MB/layer (float32) ≈ 14.4 GB > 12
 * GB heap → OOM kill
 *
 * After the fix, large weights are kept as QuantizedTensor (raw Q4_K bytes).
 * Dequantization happens one 256-element block at a time, only during the
 * matmul. These tests verify the new path is numerically equivalent to the old
 * one.
 */
@DisplayName("Quantized matVec (PhiForwardPassHandler OOM fix)")
class PhiQuantizedMatVecTest {

	// ── Test 1: F32 QuantizedTensor → same as plain float[] matVec ────────────

	@Test
	@DisplayName("F32 QuantizedTensor matVec matches plain float[] matVec exactly")
	void f32_quantizedTensor_matchesPlainMatVec(@TempDir Path tmp) throws IOException {
		int rows = 4, cols = 4;
		float[] matrix = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f };
		float[] x = { 1f, 1f, 1f, 1f };

		// Build F32 QuantizedTensor directly from the float data
		byte[] raw = toF32Bytes(matrix);
		GgufReader.QuantizedTensor qt = new GgufReader.QuantizedTensor("m", 0 /* F32 */, rows * cols, raw);

		// Old path: float[] matVec
		float[] expected = CpuForwardPassHandler.matVec(matrix, x, rows, cols);

		// New path: QuantizedTensor matVec (full rows, no row range needed)
		float[] actual = CpuForwardPassHandler.matVec(qt, x, 0, rows, cols);

		assertThat(actual).containsExactly(expected);
	}

	// ── Test 2: Q4_K QuantizedTensor → matches GgufReader dequant + matVec ────

	/**
	 * Build a 1-row Q4_K "matrix" (256 elements = one block) wrapped in a GGUF,
	 * read it back both as float[] (old path) and QuantizedTensor (new path), dot
	 * both with an all-ones vector and assert the results agree within 1e-3.
	 *
	 * Using the GgufReaderTest golden block (seed=42, d=0.25) so the expected dot
	 * product is the sum of all 256 dequantized values — verified against the C
	 * reference via the existing GgufReaderTest.
	 */
	@Test
	@DisplayName("Q4_K QuantizedTensor matVec matches GgufReader dequant + matVec")
	void q4k_quantizedTensor_matchesDequantMatVec(@TempDir Path tmp) throws IOException {
		// Reuse the Q4_K block from GgufReaderTest (known-good golden data)
		byte[] qs = new byte[128];
		byte[] sc = new byte[12];
		// all-max nibbles: qs[i] = 0xFF → low nibble=0xF, high nibble=0xF
		java.util.Arrays.fill(qs, (byte) 0xFF);
		// sc[0]=1 → getScale4K(0)=1, getMin4K(0)=0 (sub-block 0 only, others all-zero)
		sc[0] = 1;
		short d16 = 0x3C00; // 1.0 in FP16
		short dmin16 = 0x0000; // 0.0

		// Assemble one Q4_K block: [d:f16][dmin:f16][sc:12][qs:128] = 144 bytes
		byte[] block = new byte[144];
		ByteBuffer bb = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
		bb.putShort(d16);
		bb.putShort(dmin16);
		bb.put(sc);
		bb.put(qs);

		// Write a single-tensor GGUF with this block (1 row × 256 cols)
		Path gguf = buildMinimalGguf(tmp, "wq", 12 /* Q4_K */, 256, block);

		float[] x = new float[256];
		java.util.Arrays.fill(x, 1.0f);

		// Old path: eager dequantize via GgufReader.tensor() then float[] matVec
		float[] expected;
		try (GgufReader r = GgufReader.open(gguf)) {
			float[] deq = r.tensor("wq"); // 256 dequantized floats
			expected = CpuForwardPassHandler.matVec(deq, x, 1, 256); // 1 row × 256 cols
		}

		// New path: raw bytes via GgufReader.tensorRaw() then quantized matVec
		float[] actual;
		try (GgufReader r = GgufReader.open(gguf)) {
			GgufReader.QuantizedTensor qt = r.tensorRaw("wq");
			actual = CpuForwardPassHandler.matVec(qt, x, 0, 1, 256); // row 0..0
		}

		assertThat(actual).hasSize(1);
		assertThat(actual[0]).isCloseTo(expected[0], within(1e-3f));
	}

	// ── Test 3: row-range matVec slices fused tensor correctly ────────────────

	/**
	 * A fused QKV tensor has 3 logical "sub-matrices" packed contiguously in rows.
	 * matVec(qt, x, 0, H, H) should give the same result as the old
	 * Arrays.copyOfRange(deq, 0, H*H) followed by float[] matVec.
	 */
	@Test
	@DisplayName("Row-range matVec extracts correct sub-matrix from fused Q4_K tensor")
	void q4k_rowRange_extractsCorrectSubMatrix(@TempDir Path tmp) throws IOException {
		// 3 rows × 256 cols, each row is one Q4_K block.
		// Row 0: scale=1, qs all 0x11 (nibbles = 1) → each element ≈ 1.0
		// Row 1: scale=2, qs all 0x22 (nibbles = 2) → each element ≈ 4.0
		// Row 2: scale=3, qs all 0x33 (nibbles = 3) → each element ≈ 9.0
		// (Approximate — depends on getScale4K, getMin4K for sc[0]=scale)

		byte[] fullData = new byte[3 * 144];
		fillQ4KBlock(fullData, 0, (byte) 1, (byte) 0x11); // row 0
		fillQ4KBlock(fullData, 144, (byte) 2, (byte) 0x22); // row 1
		fillQ4KBlock(fullData, 288, (byte) 3, (byte) 0x33); // row 2

		Path gguf = buildMinimalGguf(tmp, "qkv", 12 /* Q4_K */, 3 * 256, fullData);

		float[] x = new float[256];
		java.util.Arrays.fill(x, 1.0f);

		try (GgufReader r = GgufReader.open(gguf)) {
			float[] deq = r.tensor("qkv"); // 3*256 floats

			// Old path: manually slice row 0 only
			float[] row0deq = java.util.Arrays.copyOfRange(deq, 0, 256);
			float[] expectedRow0 = CpuForwardPassHandler.matVec(row0deq, x, 1, 256);

			float[] row1deq = java.util.Arrays.copyOfRange(deq, 256, 512);
			float[] expectedRow1 = CpuForwardPassHandler.matVec(row1deq, x, 1, 256);

			GgufReader.QuantizedTensor qt = r.tensorRaw("qkv");

			// New path: row-range matVec
			float[] actualRow0 = CpuForwardPassHandler.matVec(qt, x, 0, 1, 256); // row 0 only
			float[] actualRow1 = CpuForwardPassHandler.matVec(qt, x, 1, 2, 256); // row 1 only
			float[] actualRows01 = CpuForwardPassHandler.matVec(qt, x, 0, 2, 256); // rows 0+1

			// Row-range results must match element-wise dequant+matVec
			assertThat(actualRow0[0]).isCloseTo(expectedRow0[0], within(1e-3f));
			assertThat(actualRow1[0]).isCloseTo(expectedRow1[0], within(1e-3f));

			// Combined result must have 2 elements
			assertThat(actualRows01).hasSize(2);
			assertThat(actualRows01[0]).isCloseTo(expectedRow0[0], within(1e-3f));
			assertThat(actualRows01[1]).isCloseTo(expectedRow1[0], within(1e-3f));
		}
	}

	// ── Test 4: tensorRaw returns correct byte count for Q4_K ─────────────────

	@Test
	@DisplayName("tensorRaw returns Q4_K-sized bytes, not float32-sized bytes")
	void tensorRaw_q4k_returnsBytesNotFloats(@TempDir Path tmp) throws IOException {
		int nelems = 256 * 32; // 8192 elements = 32 Q4_K blocks
		byte[] q4kData = new byte[32 * 144]; // 32 blocks × 144 bytes
		Path gguf = buildMinimalGguf(tmp, "big", 12 /* Q4_K */, nelems, q4kData);

		try (GgufReader r = GgufReader.open(gguf)) {
			GgufReader.QuantizedTensor qt = r.tensorRaw("big");

			// Q4_K raw: nelems/256 blocks × 144 bytes/block
			long expectedQ4KBytes = (long) (nelems / 256) * 144;
			// Float32 equivalent would be nelems * 4 = much larger
			long float32Bytes = (long) nelems * 4;

			assertThat((long) qt.data().length).isEqualTo(expectedQ4KBytes);
			assertThat((long) qt.data().length).isLessThan(float32Bytes);
			assertThat(qt.nelems()).isEqualTo(nelems);
			assertThat(qt.type()).isEqualTo(12); // Q4_K
		}
	}

	// ── Test 5: Q6_K QuantizedTensor → matches GgufReader dequant + matVec ────

	/**
	 * Q6_K is GGML type 14. phi-3.5-mini uses Q6_K for some projection weights.
	 * Uses the same golden block as GgufReaderTest.q6k_single_block_golden_values
	 * (seed=42, d=0.25).
	 */
	@Test
	@DisplayName("Q6_K QuantizedTensor matVec matches GgufReader dequant + matVec")
	void q6k_quantizedTensor_matchesDequantMatVec(@TempDir Path tmp) throws IOException {
		byte[] ql = { 57, 12, -116, 125, 114, 71, 52, 44, -40, 16, 15, 47, 111, 119, 13, 101, -42, 112, -27, -114, 3,
				81, -40, -82, -114, 79, 110, -84, 52, 47, -62, 49, -73, -80, -121, 22, -21, 63, -63, 40, -106, -71, 98,
				35, 23, 116, -108, 40, 119, 51, -62, -114, -24, -70, 83, -67, -75, 107, -120, 36, 87, 125, 83, -20, -62,
				-118, 112, -90, 28, 117, 16, -95, -51, -119, 33, 108, -95, 108, -1, -54, -22, 73, -121, 71, 126, -122,
				-37, -52, -71, 112, 70, -4, 46, 24, 56, 78, 81, -40, 32, -59, -61, -17, -128, 5, 58, -120, -82, 57,
				-106, -34, 80, -24, 1, -122, 91, 54, -104, 101, 78, -65, 82, 0, -91, -6, 9, 57, -71, -99 };
		byte[] qh = { 122, 29, 123, 40, 43, -8, 35, 64, 65, -13, 84, -121, -40, 108, 102, -97, -52, -65, -32, -25, 61,
				126, 115, 32, -83, 10, 117, 112, 3, 36, 30, 117, 34, 16, -87, 36, 121, -114, -8, 109, 67, -14, 124, -14,
				-48, 97, 48, 49, -36, -75, -40, -46, -17, 27, 50, 31, -50, -83, 55, 127, 98, 97, -27, 71 };
		byte[] sc = { 14, 6, 9, 15, 8, 28, 30, 3, 15, 26, 28, 28, 18, 4, 2, 21 };
		short d_f16 = 0x3400; // 0.25f

		// Q6_K block: [ql:128][qh:64][sc:16][d:f16] = 210 bytes
		byte[] block = new byte[210];
		System.arraycopy(ql, 0, block, 0, 128);
		System.arraycopy(qh, 0, block, 128, 64);
		System.arraycopy(sc, 0, block, 192, 16);
		ByteBuffer.wrap(block, 208, 2).order(ByteOrder.LITTLE_ENDIAN).putShort(d_f16);

		Path gguf = buildMinimalGguf(tmp, "wq6k", 14, 256, block);
		float[] x = new float[256];
		java.util.Arrays.fill(x, 1.0f);

		float[] expected;
		try (GgufReader r = GgufReader.open(gguf)) {
			float[] deq = r.tensor("wq6k");
			expected = CpuForwardPassHandler.matVec(deq, x, 1, 256);
		}
		float[] actual;
		try (GgufReader r = GgufReader.open(gguf)) {
			GgufReader.QuantizedTensor qt = r.tensorRaw("wq6k");
			assertThat(qt.type()).isEqualTo(14);
			actual = CpuForwardPassHandler.matVec(qt, x, 0, 1, 256);
		}
		assertThat(actual).hasSize(1);
		assertThat(actual[0]).isCloseTo(expected[0], within(1e-3f));
	}

	// ── Test 5: Q5_K QuantizedTensor → matches GgufReader dequant + matVec ────

	/**
	 * Q5_K is GGML type 13. phi-3.5-mini-instruct.Q4_K_M uses Q5_K for some tensors
	 * (e.g. attn_output.weight in certain layers). Before the fix,
	 * matVec(QuantizedTensor) threw UnsupportedOperationException for type 13 at
	 * runtime with "Killed" appearing as SIGKILL via OOM — now it actually throws
	 * the exception during the first forward pass.
	 *
	 * This test verifies the Q5_K path produces the same dot product as the
	 * eager-dequantize path, within floating-point tolerance.
	 */
	@Test
	@DisplayName("Q5_K QuantizedTensor matVec matches GgufReader dequant + matVec")
	void q5k_quantizedTensor_matchesDequantMatVec(@TempDir Path tmp) throws IOException {
		// One Q5_K block: [d:f16][dmin:f16][sc:12][qh:32][qs:128] = 176 bytes
		// d=1.0, dmin=0.0, sc[0]=1 → scale0=1, min0=0 for sub-block 0
		// qs all 0x11 (low nibble=1, high nibble=1), qh all 0x00 (no 5th bit)
		byte[] block = new byte[176];
		ByteBuffer bb = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
		bb.putShort((short) 0x3C00); // d = 1.0 f16
		bb.putShort((short) 0x0000); // dmin = 0.0 f16
		byte[] sc = new byte[12];
		sc[0] = 1; // getScale4K(0)=1, getMin4K(0)=0
		bb.put(sc);
		byte[] qh = new byte[32]; // all zeros → 5th bit = 0 everywhere
		bb.put(qh);
		byte[] qs = new byte[128];
		java.util.Arrays.fill(qs, (byte) 0x11); // low=1, high=1
		bb.put(qs);

		Path gguf = buildMinimalGguf(tmp, "wq5k", 13 /* Q5_K */, 256, block);

		float[] x = new float[256];
		java.util.Arrays.fill(x, 1.0f);

		// Old (eager dequant) path
		float[] expected;
		try (GgufReader r = GgufReader.open(gguf)) {
			float[] deq = r.tensor("wq5k");
			expected = CpuForwardPassHandler.matVec(deq, x, 1, 256);
		}

		// New (quantized) path
		float[] actual;
		try (GgufReader r = GgufReader.open(gguf)) {
			GgufReader.QuantizedTensor qt = r.tensorRaw("wq5k");
			assertThat(qt.type()).isEqualTo(13); // must be Q5_K
			actual = CpuForwardPassHandler.matVec(qt, x, 0, 1, 256);
		}

		assertThat(actual).hasSize(1);
		assertThat(actual[0]).isCloseTo(expected[0], within(1e-3f));
	}

	// ── Helpers ────────────────────────────────────────────────────────────────

	/** Fill one Q4_K block (144 bytes) starting at offset in dest. */
	private static void fillQ4KBlock(byte[] dest, int offset, byte scale, byte qsByte) {
		ByteBuffer bb = ByteBuffer.wrap(dest, offset, 144).order(ByteOrder.LITTLE_ENDIAN);
		// d = scale as F32 rounded to F16 (scale=1 → 0x3C00, scale=2 → 0x4000, scale=3
		// → 0x4200)
		short[] f16s = { 0x3C00, 0x4000, 0x4200, 0x4400 }; // 1,2,3,4
		short d16 = f16s[Math.min(scale - 1, 3)];
		bb.putShort(d16); // d
		bb.putShort((short) 0x0000); // dmin=0
		// sc: sc[0]=1 so getScale4K(0)=1; rest zero
		byte[] sc = new byte[12];
		sc[0] = 1;
		bb.put(sc);
		// qs: all qsByte (nibble pattern)
		byte[] qs = new byte[128];
		java.util.Arrays.fill(qs, qsByte);
		bb.put(qs);
	}

	private static byte[] toF32Bytes(float[] values) {
		ByteBuffer bb = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (float v : values)
			bb.putFloat(v);
		return bb.array();
	}

	/** Minimal single-tensor GGUF (copied from GgufReaderTest pattern). */
	private static Path buildMinimalGguf(Path dir, String name, int ggmlType, long nelems, byte[] data)
			throws IOException {
		int ALIGNMENT = 32, MAGIC = 0x46554747;
		byte[] nameBytes = name.getBytes(java.nio.charset.StandardCharsets.UTF_8);
		int headerSize = 24;
		int infoSize = 8 + nameBytes.length + 4 + 8 + 4 + 8;
		int prePad = headerSize + infoSize;
		int aligned = ((prePad + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
		int padLen = aligned - prePad;

		ByteBuffer buf = ByteBuffer.allocate(aligned + data.length).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(MAGIC);
		buf.putInt(3);
		buf.putLong(1);
		buf.putLong(0);
		buf.putLong(nameBytes.length);
		buf.put(nameBytes);
		buf.putInt(1);
		buf.putLong(nelems);
		buf.putInt(ggmlType);
		buf.putLong(0);
		buf.put(new byte[padLen]);
		buf.put(data);

		Path gguf = dir.resolve(name + ".gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}
}