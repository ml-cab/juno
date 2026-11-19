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
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("DenseKvTensor")
class DenseKvTensorTest {

	@Test
	@DisplayName("F32 write is bit-identical in the attention rows, which are the backing array")
	void f32_bit_identical() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F32, 8);
		float[] tok = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
		t.writeToken(0, tok);
		float[] view = t.attentionRows(1).floatPage(0);
		assertThat(view).startsWith(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
		assertThat(t.allocatedBytes()).isEqualTo(DenseKvTensor.INITIAL_SEQ_CAPACITY * 8 * Float.BYTES);
	}

	@Test
	@DisplayName("Q8_0 uses less persistent memory than F32 and roundtrips")
	void q8_smaller_and_roundtrips() {
		int kvDim = 256;
		int tokens = 128;
		DenseKvTensor f16 = new DenseKvTensor(KvElementType.F32, kvDim, tokens);
		DenseKvTensor q8 = new DenseKvTensor(KvElementType.Q8_0, kvDim, tokens);
		assertThat(q8.allocatedBytes() * 2L).isLessThanOrEqualTo(f16.allocatedBytes());

		float[] tok = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			tok[i] = (float) Math.sin(i * 0.01);
		q8.writeToken(0, tok);
		float[] got = q8.toFloatArray(1);
		for (int i = 0; i < kvDim; i++)
			assertThat(got[i]).isCloseTo(tok[i], within(0.02f));
	}

	@Test
	@DisplayName("Q4_0 uses about a seventh of F32's persistent memory and roundtrips within one step")
	void q4_smaller_and_roundtrips() {
		int kvDim = 256;
		int tokens = 128;
		DenseKvTensor f16 = new DenseKvTensor(KvElementType.F32, kvDim, tokens);
		DenseKvTensor q4 = new DenseKvTensor(KvElementType.Q4_0, kvDim, tokens);
		assertThat(q4.allocatedBytes()).isEqualTo(tokens * Q4_0KvCodec.encodedBytes(kvDim));
		assertThat(q4.allocatedBytes() * 7L).isLessThanOrEqualTo(f16.allocatedBytes());
		assertThat(q4.bytesPerToken()).isEqualTo(8 * 18);

		float[] tok = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			tok[i] = (float) Math.sin(i * 0.01);
		q4.writeToken(3, tok);
		float[] got = new float[kvDim];
		q4.readToken(3, got);
		for (int i = 0; i < kvDim; i++)
			assertThat(got[i]).isCloseTo(tok[i], within(1f / 8f));
		float[] unwritten = q4.toFloatArray(4);
		for (int i = 0; i < kvDim; i++)
			assertThat(unwritten[i] == 0f).as("row 0 never written reads zero at %d: %s", i, unwritten[i]).isTrue();
	}

	@Test
	@DisplayName("loadFloatPrefix + toFloatArray roundtrip for Q8_0")
	void load_prefix() {
		int kvDim = 32;
		DenseKvTensor t = new DenseKvTensor(KvElementType.Q8_0, kvDim);
		float[] prefix = new float[2 * kvDim];
		for (int i = 0; i < prefix.length; i++)
			prefix[i] = i * 0.01f;
		t.loadFloatPrefix(prefix, 2);
		float[] out = t.toFloatArray(2);
		for (int i = 0; i < prefix.length; i++)
			assertThat(out[i]).isCloseTo(prefix[i], within(0.05f));
	}
}
