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
	@DisplayName("preferPacked requires tryMmq and a K-quant with a fused kernel")
	void preferPacked_gates() {
		var q4 = new GgufReader.QuantizedTensor("q4", QuantizationLayout.TYPE_Q4_K, 256, new byte[144]);
		var q5 = new GgufReader.QuantizedTensor("q5", QuantizationLayout.TYPE_Q5_K, 256, new byte[176]);
		var q6 = new GgufReader.QuantizedTensor("q6", QuantizationLayout.TYPE_Q6_K, 256, new byte[210]);
		var q8 = new GgufReader.QuantizedTensor("q8", 8 /* Q8_0 */, 32, new byte[34]);

		assertThat(Q4KResidentUpload.preferPacked(false, q4)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, null)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, q8)).isFalse();
		assertThat(Q4KResidentUpload.preferPacked(true, q4)).isTrue();
		assertThat(Q4KResidentUpload.preferPacked(true, q5)).isTrue();
		assertThat(Q4KResidentUpload.preferPacked(true, q6)).isTrue();
	}

	@Test
	@DisplayName("Q6_K blocks are re-slotted to 224-byte aligned slots with zero pad")
	void q6k_padding_reslots_blocks() {
		int blockBytes = QuantizationLayout.Q6_K.blockBytes();
		byte[] raw = new byte[3 * blockBytes];
		for (int i = 0; i < raw.length; i++)
			raw[i] = (byte) (i * 7 + 1);

		byte[] padded = DeviceQ4KMatrix.padQ6KBlocks(raw);

		assertThat(padded).hasSize(3 * DeviceQ4KMatrix.Q6K_SLOT_BYTES);
		for (int b = 0; b < 3; b++) {
			int src = b * blockBytes;
			int dst = b * DeviceQ4KMatrix.Q6K_SLOT_BYTES;
			for (int i = 0; i < blockBytes; i++)
				assertThat(padded[dst + i]).as("block " + b + " byte " + i).isEqualTo(raw[src + i]);
			for (int i = blockBytes; i < DeviceQ4KMatrix.Q6K_SLOT_BYTES; i++)
				assertThat(padded[dst + i]).as("block " + b + " pad " + i).isZero();
		}
	}

	@Test
	@DisplayName("Q8_1 scratch is 36 bytes per 32 activations")
	void q8Bytes_layout() {
		assertThat(Q4KMmqKernel.q8Bytes(256)).isEqualTo(8L * 36);
		assertThat(Q4KMmqKernel.q8Bytes(3072)).isEqualTo(96L * 36);
	}

	@Test
	@DisplayName("supportsType covers exactly Q4_K / Q5_K / Q6_K")
	void supportsType_set() {
		assertThat(DeviceQ4KMatrix.supportsType(QuantizationLayout.TYPE_Q4_K)).isTrue();
		assertThat(DeviceQ4KMatrix.supportsType(QuantizationLayout.TYPE_Q5_K)).isTrue();
		assertThat(DeviceQ4KMatrix.supportsType(QuantizationLayout.TYPE_Q6_K)).isTrue();
		assertThat(DeviceQ4KMatrix.supportsType(8)).isFalse();
		assertThat(DeviceQ4KMatrix.supportsType(0)).isFalse();
	}
}
