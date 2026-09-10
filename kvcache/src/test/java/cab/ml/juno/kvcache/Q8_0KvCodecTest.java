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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Q8_0KvCodec")
class Q8_0KvCodecTest {

	@Test
	@DisplayName("encodedBytes pads to QK blocks")
	void encoded_bytes_pads() {
		assertThat(Q8_0KvCodec.encodedBytes(0)).isEqualTo(0);
		assertThat(Q8_0KvCodec.encodedBytes(1)).isEqualTo(Q8_0KvCodec.BLOCK_BYTES);
		assertThat(Q8_0KvCodec.encodedBytes(32)).isEqualTo(Q8_0KvCodec.BLOCK_BYTES);
		assertThat(Q8_0KvCodec.encodedBytes(33)).isEqualTo(2 * Q8_0KvCodec.BLOCK_BYTES);
	}

	@Test
	@DisplayName("roundtrip keeps small abs error on unit-scale vector")
	void roundtrip_unit_scale() {
		float[] src = new float[64];
		for (int i = 0; i < src.length; i++)
			src[i] = (i - 32) / 32f;
		assertThat(Q8_0KvCodec.maxAbsError(src, 0, src.length)).isLessThan(0.01f);
	}

	@Test
	@DisplayName("roundtrip preserves zeros and handles non-multiple length")
	void roundtrip_partial_block() {
		float[] src = { 0f, 0.5f, -0.25f, 1f, -1f };
		byte[] enc = new byte[Q8_0KvCodec.encodedBytes(src.length)];
		Q8_0KvCodec.encode(src, 0, src.length, enc, 0);
		float[] got = new float[src.length];
		Q8_0KvCodec.decode(enc, 0, got, 0, src.length);
		for (int i = 0; i < src.length; i++)
			assertThat(got[i]).isCloseTo(src[i], within(0.02f));
	}

	@Test
	@DisplayName("compression vs float32 is at least 2x for typical kvDim")
	void compression_vs_f32() {
		assertThat(Q8_0KvCodec.compressionRatioVsF32(256)).isGreaterThanOrEqualTo(2.0);
		assertThat(Q8_0KvCodec.compressionRatioVsF32(1024 * 256)).isGreaterThanOrEqualTo(2.0);
	}

	@Test
	@DisplayName("rejects bad ranges")
	void rejects_bad_ranges() {
		float[] src = new float[8];
		byte[] dst = new byte[Q8_0KvCodec.BLOCK_BYTES];
		assertThatThrownBy(() -> Q8_0KvCodec.encode(src, 0, 16, dst, 0))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
