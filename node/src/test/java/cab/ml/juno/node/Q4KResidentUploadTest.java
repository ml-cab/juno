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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Q4KResidentUpload policy")
class Q4KResidentUploadTest {

	@Test
	@DisplayName("preferPacked requires tryMmq and TYPE_Q4_K")
	void preferPacked_gates() {
		var q4 = new GgufReader.QuantizedTensor("q4", QuantizationLayout.TYPE_Q4_K, 256, new byte[144]);
		var q8 = new GgufReader.QuantizedTensor("q8", 8 /* Q8_0 */, 32, new byte[34]);

		assertThat(Q4KResidentUpload.preferPacked(false, q4)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, null)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, q8)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, q4)).isTrue();
	}
}
