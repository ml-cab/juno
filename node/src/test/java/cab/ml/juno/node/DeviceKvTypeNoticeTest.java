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

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CacheTypeOptions;
import cab.ml.juno.kvcache.KvElementType;

/**
 * GPU attention reads a device copy of the KV that is always FP16, so a quantized {@code --cache-type-k/-v} on a launch
 * that attends on the GPU sets the host rows only. That launch is told so at startup; every launch where attention
 * reads the host rows (CPU backend, a GPU backend without the kernel, LoRA, the CPU-only handlers,
 * {@code --gpu-attention off}) is not.
 */
@DisplayName("DeviceKvTypeNotice")
class DeviceKvTypeNoticeTest {

	private static final CacheTypeOptions F16 = CacheTypeOptions.defaults();
	private static final CacheTypeOptions Q4_K_ONLY = CacheTypeOptions.of(KvElementType.Q4_0, KvElementType.F16);
	private static final CacheTypeOptions Q8 = CacheTypeOptions.of(KvElementType.Q8_0, KvElementType.Q8_0);

	private String prevAttention;

	@BeforeEach
	void saveAttention() {
		prevAttention = System.getProperty(GpuAttentionOptions.ENV_PROPERTY);
		System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
	}

	@AfterEach
	void restoreAttention() {
		if (prevAttention == null)
			System.clearProperty(GpuAttentionOptions.ENV_PROPERTY);
		else
			System.setProperty(GpuAttentionOptions.ENV_PROPERTY, prevAttention);
	}

	@Test
	@DisplayName("a CUDA launch with GPU attention and a quantized cache type is told the device copy stays FP16")
	void quantizedTypeOnGpuAttentionGetsTheNotice() {
		String notice = DeviceKvTypeNotice.consoleNotice("llama", false, false, true, Q4_K_ONLY);
		assertThat(notice).isNotNull().contains("cache-type-k=q4_0 cache-type-v=f16").contains("host KV")
				.contains("FP16").doesNotContain("\n");
		assertThat(DeviceKvTypeNotice.consoleNotice("phi3", false, false, true, Q8)).contains("cache-type-k=q8_0");
		assertThat(DeviceKvTypeNotice.consoleNotice("qwen3", false, false, true, Q8)).isNotNull();
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "on");
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", false, false, true, Q8)).isNotNull();
	}

	@Test
	@DisplayName("no notice where attention reads the host rows, or the cache type is f16")
	void noNoticeWhereAttentionReadsTheHostRows() {
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", false, false, true, F16)).as("f16").isNull();
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", false, true, false, Q4_K_ONLY)).as("CPU").isNull();
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", false, false, false, Q4_K_ONLY)).as("ROCm").isNull();
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", true, false, true, Q4_K_ONLY)).as("LoRA").isNull();
		assertThat(DeviceKvTypeNotice.consoleNotice("phi2", false, false, true, Q4_K_ONLY)).as("phi2").isNull();
		assertThat(DeviceKvTypeNotice.consoleNotice("qwen3moe", false, false, true, Q4_K_ONLY)).as("qwen3moe")
				.isNull();
		System.setProperty(GpuAttentionOptions.ENV_PROPERTY, "off");
		assertThat(DeviceKvTypeNotice.consoleNotice("llama", false, false, true, Q4_K_ONLY)).as("off").isNull();
	}

	@Test
	@DisplayName("a handler that opens GPU attention logs it once per process, and nothing for f16")
	void handlerLogsOnce() {
		Logger log = Logger.getLogger(DeviceKvTypeNotice.class.getName());
		List<LogRecord> records = new ArrayList<>();
		Handler capture = new Handler() {
			@Override
			public void publish(LogRecord record) {
				records.add(record);
			}

			@Override
			public void flush() {
			}

			@Override
			public void close() {
			}
		};
		log.addHandler(capture);
		try {
			DeviceKvTypeNotice.resetForTest();
			DeviceKvTypeNotice.announce("Llama", F16);
			assertThat(records).isEmpty();
			DeviceKvTypeNotice.announce("Llama", Q4_K_ONLY);
			DeviceKvTypeNotice.announce("Phi-3", Q4_K_ONLY);
			assertThat(records).singleElement().satisfies(r -> {
				assertThat(r.getLevel()).isEqualTo(Level.WARNING);
				assertThat(r.getMessage()).contains("Llama").contains("cache-type-k=q4_0").contains("FP16");
			});
		} finally {
			log.removeHandler(capture);
			DeviceKvTypeNotice.resetForTest();
		}
	}
}
