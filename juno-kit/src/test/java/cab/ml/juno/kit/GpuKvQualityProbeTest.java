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
package cab.ml.juno.kit;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.node.DeviceKvTypeNotice;

/** The GPU KV quality probe's reading of the handler's device KV notice, without a GPU. */
class GpuKvQualityProbeTest {

	private static final String STORED_Q8 = "LlamaTransformerHandler: GPU attention's KV copy stored as K q8_0, V q8_0, "
			+ "8704 bytes per position against 16384 for FP16";

	/** The handler's own FP16 warning text, for a head width the quantized blocks cannot hold. */
	private static final String KEPT_FP16 = DeviceKvTypeNotice.consoleNotice("llama", 16, false, false, true,
			CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q8_0));

	@Test
	void theStoredLineForTheSelectedFormatsPasses() {
		ProbeReport.Check c = GpuKvQualityProbe.storedCheck("q8_0", STORED_Q8);
		assertThat(c.passed()).isTrue();
		assertThat(c.name()).isEqualTo("q8_0 device KV stored in the selected formats");
		assertThat(c.detail()).isEqualTo(STORED_Q8);
	}

	@Test
	void anotherFormatsStoredLineFails() {
		assertThat(GpuKvQualityProbe.storedCheck("q8_0/q4_0", STORED_Q8).passed()).isFalse();
	}

	@Test
	void theFp16WarningFailsAndIsTheDetail() {
		assertThat(KEPT_FP16).contains("head width 16");
		ProbeReport.Check c = GpuKvQualityProbe.storedCheck("q8_0", KEPT_FP16);
		assertThat(c.passed()).isFalse();
		assertThat(c.detail()).isEqualTo(KEPT_FP16);
	}

	@Test
	void noNoticeFailsSayingGpuAttentionDidNotOpen() {
		ProbeReport.Check c = GpuKvQualityProbe.storedCheck("q8_0", null);
		assertThat(c.passed()).isFalse();
		assertThat(c.detail()).contains("did not open");
	}

	@Test
	void theNoticeIsPickedFromTheRunsMessages() {
		assertThat(GpuKvQualityProbe.deviceKvNotice(List.of("GPU-resident decode region active", STORED_Q8)))
				.isEqualTo(STORED_Q8);
		assertThat(GpuKvQualityProbe.deviceKvNotice(List.of("model loaded", KEPT_FP16))).isEqualTo(KEPT_FP16);
		assertThat(GpuKvQualityProbe.deviceKvNotice(List.of("model loaded", "GPU attention kernel loaded"))).isNull();
	}
}
