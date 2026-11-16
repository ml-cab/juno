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
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.node.DeviceKvTypeNotice;

/** The GPU KV prefill probe's reading of its runs, without a model or a GPU. */
class GpuKvPrefillProbeTest {

	private static final String STORED_Q8 = "LlamaTransformerHandler: GPU attention's KV copy stored as K q8_0, V q8_0, "
			+ "8704 bytes per position against 16384 for FP16";

	private static final String KEPT_FP16 = DeviceKvTypeNotice.consoleNotice("llama", 16, false, false, true,
			CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q8_0));

	@Test
	void theRunsLineIsParsedWithItsWidthsInOrder() throws IOException {
		String out = "loading\n" + KvPrefillRunMain.PREFIX
				+ "{\"ms\":{\"2048\":[1000.0,990.0,1010.0],\"128\":[1300.0,1290.0,1310.0]},\"gpuAttention\":true,"
				+ "\"prefillRegion\":true,\"kv\":\"K q8_0, V q8_0\",\"deviceKv\":\"" + STORED_Q8 + "\"}\nbye\n";
		GpuKvPrefillProbe.Run run = GpuKvPrefillProbe.parse(out, "q8_0", 0);
		assertThat(run.ms().keySet()).containsExactly("2048", "128");
		assertThat(run.ms().get("128")).containsExactly(1300.0, 1290.0, 1310.0);
		assertThat(run.gpuAttention()).isTrue();
		assertThat(run.deviceKv()).isEqualTo(STORED_Q8);
	}

	@Test
	void aRunWithoutItsLineFailsNamingTheTypeAndExit() {
		assertThatThrownBy(() -> GpuKvPrefillProbe.parse("Exception in thread main\n", "q8_0", 1))
				.isInstanceOf(IOException.class).hasMessageContaining("q8_0").hasMessageContaining("exit 1");
	}

	@Test
	void eachWidthGivesBothSidesAndTheRatioOfMedians() {
		GpuKvPrefillProbe.Run f16 = run(Map.of("2048", List.of(1000.0, 900.0, 1100.0)), null);
		GpuKvPrefillProbe.Run q8 = run(Map.of("2048", List.of(1050.0, 990.0, 2000.0)), STORED_Q8);
		List<ProbeReport.Reading> r = GpuKvPrefillProbe.readings(f16, q8, "q8_0");
		assertThat(r).extracting(ProbeReport.Reading::name).containsExactly(
				"prefill ms, f16 device KV, windows of 2048 tokens", "prefill ms, q8_0 device KV, windows of 2048 tokens",
				"q8_0 over f16 prefill time, windows of 2048 tokens");
		assertThat(r.get(0).scored()).isTrue();
		assertThat(r.get(0).higherIsBetter()).isFalse();
		assertThat(r.get(2).values()).containsExactly(1050.0 / 1000.0);
		assertThat(r.get(2).scored()).isFalse();
	}

	@Test
	void aWidthTheQuantizedRunLacksFails() {
		GpuKvPrefillProbe.Run f16 = run(Map.of("128", List.of(1.0)), null);
		GpuKvPrefillProbe.Run q8 = run(Map.of("2048", List.of(1.0)), STORED_Q8);
		assertThatThrownBy(() -> GpuKvPrefillProbe.readings(f16, q8, "q8_0")).hasMessageContaining("windows of 128");
	}

	@Test
	void onlyADevicePrefillInTheSelectedFormatsIsRead() {
		assertThatCode(() -> GpuKvPrefillProbe.requireDevicePrefill(run(Map.of(), null), "f16")).doesNotThrowAnyException();
		assertThatCode(() -> GpuKvPrefillProbe.requireDevicePrefill(run(Map.of(), STORED_Q8), "q8_0"))
				.doesNotThrowAnyException();
		assertThatThrownBy(() -> GpuKvPrefillProbe.requireDevicePrefill(run(Map.of(), KEPT_FP16), "q8_0"))
				.hasMessageContaining("head width 16");
		assertThatThrownBy(() -> GpuKvPrefillProbe.requireDevicePrefill(run(Map.of(), null), "q8_0"))
				.hasMessageContaining("did not open");
		assertThatThrownBy(() -> GpuKvPrefillProbe.requireDevicePrefill(
				new GpuKvPrefillProbe.Run(Map.of(), false, true, "", null), "f16")).hasMessageContaining("GPU attention");
		assertThatThrownBy(() -> GpuKvPrefillProbe.requireDevicePrefill(
				new GpuKvPrefillProbe.Run(Map.of(), true, false, "", null), "f16")).hasMessageContaining("prefill region");
	}

	private static GpuKvPrefillProbe.Run run(Map<String, List<Double>> ms, String notice) {
		return new GpuKvPrefillProbe.Run(new LinkedHashMap<>(ms), true, true, "", notice);
	}
}
