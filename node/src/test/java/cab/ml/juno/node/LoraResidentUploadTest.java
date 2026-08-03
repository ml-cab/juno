/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraResidentUpload")
class LoraResidentUploadTest {

	private static final Logger LOG = Logger.getLogger(LoraResidentUploadTest.class.getName());

	private String prevMicrobatch;
	private String prevDevice;

	@BeforeEach
	void save() {
		prevMicrobatch = System.getProperty(LoraMicrobatch.PROPERTY);
		prevDevice = System.getProperty("juno.lora.train.device");
		LoraMicrobatch.apply(8);
		System.setProperty("juno.lora.train.device", LoraTrainDevice.AUTO);
	}

	@AfterEach
	void restore() {
		if (prevMicrobatch == null)
			System.clearProperty(LoraMicrobatch.PROPERTY);
		else
			System.setProperty(LoraMicrobatch.PROPERTY, prevMicrobatch);
		if (prevDevice == null)
			System.clearProperty("juno.lora.train.device");
		else
			System.setProperty("juno.lora.train.device", prevDevice);
	}

	@Test
	@DisplayName("VRAM OOM with microbatch>1 and half support retries at microbatch=1")
	void retries_fp16_then_succeeds() {
		AtomicInteger attempts = new AtomicInteger();
		List<String> closes = new ArrayList<>();
		LoraResidentUpload.run(true, LOG, () -> closes.add("close"), () -> {
			if (attempts.incrementAndGet() == 1)
				throw new IllegalStateException("cudaMalloc failed: out of memory");
		});
		assertThat(attempts.get()).isEqualTo(2);
		assertThat(closes).containsExactly("close");
		assertThat(LoraMicrobatch.current()).isEqualTo(1);
	}

	@Test
	@DisplayName("second OOM under auto falls back to CPU via tryRecoverFromUploadOom")
	void second_oom_auto_falls_to_cpu() {
		AtomicInteger attempts = new AtomicInteger();
		AtomicInteger closes = new AtomicInteger();
		LoraResidentUpload.run(true, LOG, closes::incrementAndGet, () -> {
			attempts.incrementAndGet();
			throw new IllegalStateException("hipMalloc failed: out of memory");
		});
		assertThat(attempts.get()).isEqualTo(2);
		assertThat(closes.get()).isEqualTo(2); // retry closer + recover closer
		assertThat(LoraMicrobatch.current()).isEqualTo(1);
	}

	@Test
	@DisplayName("second OOM under gpu fails closed after FP16 retry")
	void second_oom_gpu_fails_closed() {
		System.setProperty("juno.lora.train.device", LoraTrainDevice.GPU);
		assertThatThrownBy(() -> LoraResidentUpload.run(true, LOG, () -> {
		}, () -> {
			throw new IllegalStateException("cudaMalloc failed: out of memory");
		})).isInstanceOf(IllegalStateException.class).hasMessageContaining("--lora-train-device=gpu");
		assertThat(LoraMicrobatch.current()).isEqualTo(1);
	}

	@Test
	@DisplayName("no FP16 retry when microbatch already 1")
	void no_retry_when_microbatch_one() {
		LoraMicrobatch.apply(1);
		AtomicInteger attempts = new AtomicInteger();
		LoraResidentUpload.run(true, LOG, () -> {
		}, () -> {
			attempts.incrementAndGet();
			throw new IllegalStateException("cudaMalloc failed: out of memory");
		});
		assertThat(attempts.get()).isEqualTo(1);
	}

	@Test
	@DisplayName("no FP16 retry when half residency unsupported")
	void no_retry_without_half() {
		AtomicInteger attempts = new AtomicInteger();
		LoraResidentUpload.run(false, LOG, () -> {
		}, () -> {
			attempts.incrementAndGet();
			throw new IllegalStateException("cudaMalloc failed: out of memory");
		});
		assertThat(attempts.get()).isEqualTo(1);
		assertThat(LoraMicrobatch.current()).isEqualTo(8);
	}

	@Test
	@DisplayName("non-VRAM errors are rethrown without retry")
	void non_vram_rethrows() {
		AtomicInteger attempts = new AtomicInteger();
		assertThatThrownBy(() -> LoraResidentUpload.run(true, LOG, () -> {
		}, () -> {
			attempts.incrementAndGet();
			throw new IllegalStateException("cublasCreate failed");
		})).isInstanceOf(IllegalStateException.class).hasMessageContaining("cublasCreate");
		assertThat(attempts.get()).isEqualTo(1);
		assertThat(LoraMicrobatch.current()).isEqualTo(8);
	}
}
