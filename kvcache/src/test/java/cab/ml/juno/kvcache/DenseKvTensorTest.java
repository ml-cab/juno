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
	@DisplayName("F16 write is bit-identical via viewForAttention")
	void f16_bit_identical() {
		DenseKvTensor t = new DenseKvTensor(KvElementType.F16, 8);
		float[] tok = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
		t.writeToken(0, tok);
		float[] view = t.viewForAttention(1, null);
		assertThat(view).startsWith(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
		assertThat(t.allocatedBytes()).isEqualTo(DenseKvTensor.INITIAL_SEQ_CAPACITY * 8 * Float.BYTES);
	}

	@Test
	@DisplayName("Q8_0 uses less persistent memory than F16 and roundtrips")
	void q8_smaller_and_roundtrips() {
		int kvDim = 256;
		int tokens = 128;
		DenseKvTensor f16 = new DenseKvTensor(KvElementType.F16, kvDim, tokens);
		DenseKvTensor q8 = new DenseKvTensor(KvElementType.Q8_0, kvDim, tokens);
		assertThat(q8.allocatedBytes() * 2L).isLessThanOrEqualTo(f16.allocatedBytes());

		float[] tok = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			tok[i] = (float) Math.sin(i * 0.01);
		q8.writeToken(0, tok);
		float[] scratch = new float[kvDim];
		float[] got = q8.viewForAttention(1, scratch);
		for (int i = 0; i < kvDim; i++)
			assertThat(got[i]).isCloseTo(tok[i], within(0.02f));
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
