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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Tier-5 Gate A: shared K-quant codec conformance.
 *
 * <p>Decode goldens are pinned to the same llama.cpp-derived Q6_K vectors as
 * {@link GgufReaderTest}. Encoder strategy is {@link GgufKQuantCodec#ENCODER_ID}.
 */
@DisplayName("GgufKQuantCodec / GgufQuantCodec (Tier-5 Gate A)")
class GgufKQuantCodecTest {

	@Test
	@DisplayName("QuantizationLayout exposes Q4_K/Q5_K/Q6_K geometry")
	void layout_geometry() {
		assertThat(QuantizationLayout.Q4_K.blockWidth()).isEqualTo(256);
		assertThat(QuantizationLayout.Q4_K.subBlockWidth()).isEqualTo(32);
		assertThat(QuantizationLayout.Q4_K.blockBytes()).isEqualTo(144);
		assertThat(QuantizationLayout.Q4_K.affine()).isTrue();

		assertThat(QuantizationLayout.Q5_K.blockBytes()).isEqualTo(176);
		assertThat(QuantizationLayout.Q5_K.affine()).isTrue();

		assertThat(QuantizationLayout.Q6_K.subBlockWidth()).isEqualTo(16);
		assertThat(QuantizationLayout.Q6_K.blockBytes()).isEqualTo(210);
		assertThat(QuantizationLayout.Q6_K.symmetric()).isTrue();
		assertThat(QuantizationLayout.Q6_K.affine()).isFalse();

		assertThat(QuantizationLayout.Q4_K.encodedBytes(512)).isEqualTo(288);
		assertThat(GgufQuantCodec.encoderId(12)).isEqualTo(GgufKQuantCodec.ENCODER_ID);
	}

	@Test
	@DisplayName("layout validation rejects partial blocks and bad dims")
	void layout_rejects_malformed() {
		assertThatThrownBy(() -> QuantizationLayout.Q4_K.validateElementCount(255))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("not divisible");
		assertThatThrownBy(() -> QuantizationLayout.Q6_K.validateMatrix(2, 128))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("cols");
		assertThatThrownBy(() -> GgufKQuantCodec.decode(new byte[100], 12))
				.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("Q6_K decode matches pinned llama.cpp golden values (seed=42)")
	void q6k_decode_matches_llama_cpp_golden() {
		byte[] ql = { 57, 12, -116, 125, 114, 71, 52, 44, -40, 16, 15, 47, 111, 119, 13, 101, -42, 112, -27, -114, 3,
				81, -40, -82, -114, 79, 110, -84, 52, 47, -62, 49, -73, -80, -121, 22, -21, 63, -63, 40, -106, -71, 98,
				35, 23, 116, -108, 40, 119, 51, -62, -114, -24, -70, 83, -67, -75, 107, -120, 36, 87, 125, 83, -20, -62,
				-118, 112, -90, 28, 117, 16, -95, -51, -119, 33, 108, -95, 108, -1, -54, -22, 73, -121, 71, 126, -122,
				-37, -52, -71, 112, 70, -4, 46, 24, 56, 78, 81, -40, 32, -59, -61, -17, -128, 5, 58, -120, -82, 57,
				-106, -34, 80, -24, 1, -122, 91, 54, -104, 101, 78, -65, 82, 0, -91, -6, 9, 57, -71, -99, };
		byte[] qh = { 122, 29, 123, 40, 43, -8, 35, 64, 65, -13, 84, -121, -40, 108, 102, -97, -52, -65, -32, -25, 61,
				126, 115, 32, -83, 10, 117, 112, 3, 36, 30, 117, 34, 16, -87, 36, 121, -114, -8, 109, 67, -14, 124, -14,
				-48, 97, 48, 49, -36, -75, -40, -46, -17, 27, 50, 31, -50, -83, 55, 127, 98, 97, -27, 71, };
		byte[] sc = { 14, 6, 9, 15, 8, 28, 30, 3, 15, 26, 28, 28, 18, 4, 2, 21 };
		short d_f16 = 0x3400; // 0.25

		ByteBuffer buf = ByteBuffer.allocate(210).order(ByteOrder.LITTLE_ENDIAN);
		buf.put(ql).put(qh).put(sc).putShort(d_f16);
		float[] actual = GgufQuantCodec.decode(buf.array(), QuantizationLayout.TYPE_Q6_K);

		float eps = 0.001f;
		assertThat(actual[0]).isCloseTo(31.5000f, within(eps));
		assertThat(actual[1]).isCloseTo(-14.0000f, within(eps));
		assertThat(actual[2]).isCloseTo(98.0000f, within(eps));
		assertThat(actual[3]).isCloseTo(-66.5000f, within(eps));
		assertThat(actual[32]).isCloseTo(15.7500f, within(eps));
		assertThat(actual[64]).isCloseTo(38.0000f, within(eps));
		assertThat(actual[96]).isCloseTo(-37.5000f, within(eps));
		assertThat(actual[128]).isCloseTo(7.5000f, within(eps));
		assertThat(actual[192]).isCloseTo(54.0000f, within(eps));
		assertThat(actual[255]).isCloseTo(-36.7500f, within(eps));
	}

	@Test
	@DisplayName("Q4_K all-zero quants decode to uniform -dmin")
	void q4k_all_zero_quants() {
		ByteBuffer buf = ByteBuffer.allocate(144).order(ByteOrder.LITTLE_ENDIAN);
		buf.putShort((short) 0x3C00); // d = 1.0
		buf.putShort((short) 0x3800); // dmin = 0.5
		byte[] sc = new byte[12];
		sc[0] = 1;
		sc[4] = 1; // scale[0]=1, min[0]=1 — enough for first sub-block path
		// Fill all scale/min slots with 1 so every sub-block uses dmin*1
		for (int j = 0; j < 4; j++) {
			sc[j] = (byte) ((1 & 0x3F) | ((1 & 0x30) << 2));
			sc[j + 4] = (byte) ((1 & 0x3F) | ((1 & 0x30) << 2));
			sc[j + 8] = (byte) ((1 & 0x0F) | ((1 & 0x0F) << 4));
		}
		buf.put(sc);
		buf.put(new byte[128]); // qs = 0
		float[] out = GgufKQuantCodec.decodeQ4K(buf.array(), 256);
		for (float v : out) {
			assertThat(v).isCloseTo(-0.5f, within(1e-5f));
		}
	}

	@Test
	@DisplayName("encode→decode roundtrip keeps RMSE bounded for each K-quant")
	void encode_decode_roundtrip_rmse() {
		Random rnd = new Random(7);
		float[] data = new float[256];
		for (int i = 0; i < data.length; i++) {
			data[i] = (rnd.nextFloat() * 2f - 1f) * 3f;
		}
		for (int type : new int[] {
				QuantizationLayout.TYPE_Q4_K,
				QuantizationLayout.TYPE_Q5_K,
				QuantizationLayout.TYPE_Q6_K }) {
			byte[] encoded = GgufQuantCodec.encode(data, type);
			float[] decoded = GgufQuantCodec.decode(encoded, type);
			QuantizedMergeMetrics m = QuantizedMergeMetrics.ofReconstruction(data, decoded);
			assertThat(m.rmse())
					.as("type %s", type)
					.isLessThan(0.5);
			assertThat(m.maxAbsError()).isLessThan(2.0);
		}
	}

	@Test
	@DisplayName("copyRawUnchanged is byte-identical (no-op merge path)")
	void no_op_raw_copy_is_byte_identical() {
		byte[] raw = new byte[144];
		new Random(1).nextBytes(raw);
		byte[] copy = GgufQuantCodec.copyRawUnchanged(raw);
		assertThat(copy).isEqualTo(raw);
		assertThat(copy).isNotSameAs(raw);
		// Contrast: decode/encode is lossy and must NOT be used for no-op.
		float[] decoded = GgufQuantCodec.decode(raw, QuantizationLayout.TYPE_Q4_K);
		byte[] reencoded = GgufQuantCodec.encode(decoded, QuantizationLayout.TYPE_Q4_K);
		// Re-encode may or may not match depending on block contents; identity is not required.
		assertThat(reencoded).hasSize(raw.length);
	}

	@Test
	@DisplayName("zero-delta projected path must use raw copy, not requantize")
	void zero_delta_preserves_bytes_via_copy_not_requant() {
		Random rnd = new Random(99);
		float[] data = new float[256];
		for (int i = 0; i < data.length; i++) data[i] = rnd.nextFloat();
		byte[] original = GgufQuantCodec.encode(data, QuantizationLayout.TYPE_Q5_K);
		byte[] preserved = GgufQuantCodec.copyRawUnchanged(original);
		assertThat(preserved).isEqualTo(original);
		assertThat(preserved).hasSize(QuantizationLayout.Q5_K.blockBytes());
	}

	@Test
	@DisplayName("Q6_K cannot absorb an arbitrary additive group shift (non-closure)")
	void q6k_additive_shift_not_closed() {
		float[] base = new float[256];
		Arrays.fill(base, 1.0f);
		byte[] enc = GgufQuantCodec.encode(base, QuantizationLayout.TYPE_Q6_K);
		float[] decoded = GgufQuantCodec.decode(enc, QuantizationLayout.TYPE_Q6_K);

		float[] shifted = Arrays.copyOf(decoded, decoded.length);
		for (int i = 0; i < shifted.length; i++) shifted[i] += 0.37f;

		byte[] reenc = GgufQuantCodec.encode(shifted, QuantizationLayout.TYPE_Q6_K);
		float[] round = GgufQuantCodec.decode(reenc, QuantizationLayout.TYPE_Q6_K);

		// Symmetric Q6_K has no zero/min term: constant shift is not absorbed exactly.
		double maxShiftError = 0;
		for (int i = 0; i < 256; i++) {
			maxShiftError = Math.max(maxShiftError, Math.abs((decoded[i] + 0.37f) - round[i]));
		}
		assertThat(maxShiftError).isGreaterThan(1e-3);
	}

	@Test
	@DisplayName("Q4_K nested scales do not exactly represent arbitrary new offsets")
	void q4k_nested_scale_non_closure() {
		float[] base = new float[256];
		for (int i = 0; i < 256; i++) base[i] = (i % 32) * 0.01f;
		byte[] enc = GgufQuantCodec.encode(base, QuantizationLayout.TYPE_Q4_K);
		float[] decoded = GgufQuantCodec.decode(enc, QuantizationLayout.TYPE_Q4_K);

		float[] shifted = Arrays.copyOf(decoded, decoded.length);
		for (int i = 0; i < 32; i++) shifted[i] += 0.11f; // constant per 32-group

		byte[] reenc = GgufQuantCodec.encode(shifted, QuantizationLayout.TYPE_Q4_K);
		float[] round = GgufQuantCodec.decode(reenc, QuantizationLayout.TYPE_Q4_K);

		double maxErr = 0;
		for (int i = 0; i < 32; i++) {
			maxErr = Math.max(maxErr, Math.abs((decoded[i] + 0.11f) - round[i]));
		}
		// Discretized nested scale/min: arbitrary offset generally not exact.
		assertThat(maxErr).isGreaterThan(0.0);
	}

	@Test
	@DisplayName("row-strided decode matches contiguous decode for one-row matrix")
	void rows_match_contiguous() {
		Random rnd = new Random(3);
		float[] data = new float[512];
		for (int i = 0; i < data.length; i++) data[i] = rnd.nextFloat() * 2 - 1;
		byte[] enc = GgufQuantCodec.encode(data, QuantizationLayout.TYPE_Q4_K);
		float[] contig = GgufQuantCodec.decode(enc, QuantizationLayout.TYPE_Q4_K);
		float[] rows = GgufQuantCodec.decodeRows(enc, QuantizationLayout.TYPE_Q4_K, 1, 512);
		assertThat(rows).containsExactly(contig);
	}

	@Test
	@DisplayName("deltaRetention is 1 for identical deltas and <1 when lost")
	void delta_retention_metric() {
		float[] target = { 1f, 2f, 3f };
		assertThat(QuantizedMergeMetrics.deltaRetention(target, target)).isEqualTo(1.0);
		float[] retained = { 1f, 2f, 0f };
		assertThat(QuantizedMergeMetrics.deltaRetention(target, retained)).isLessThan(1.0);
		assertThat(QuantizedMergeMetrics.deltaRetention(new float[] { 0f }, new float[] { 0f }))
				.isEqualTo(1.0);
	}

	@Test
	@DisplayName("LoraMerge.requantize K-quants delegates to versioned codec")
	void lora_merge_requantize_matches_codec() {
		float[] data = new float[256];
		Arrays.fill(data, 0.25f);
		byte[] viaMerge = LoraMerge.requantize(data, QuantizationLayout.TYPE_Q4_K, 256, 1);
		byte[] viaCodec = GgufQuantCodec.encode(data, QuantizationLayout.TYPE_Q4_K);
		assertThat(viaMerge).isEqualTo(viaCodec);
	}
}
