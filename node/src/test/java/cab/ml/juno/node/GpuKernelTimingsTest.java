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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link GpuKernelTimings}: the device time of single launches of the longest kernels at the sweep models'
 * shapes. The GPU cases run on a CUDA host; the refusal runs where there is none.
 */
@DisplayName("GpuKernelTimings")
class GpuKernelTimingsTest {

	@Test
	@Tag("gpu")
	@DisplayName("times every kernel at every sweep shape, one CUDA-event reading per launch")
	void timesEveryKernelAtEverySweepShape() {
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		List<GpuKernelTimings.Timing> t = GpuKernelTimings.measure(64, 3);
		assertThat(t).extracting(GpuKernelTimings.Timing::name).containsExactly(
				"prefill attention, tinyllama-1.1b 32/4/64 heads, 64 rows over up to 64 keys",
				"prefill attention, qwen2.5-3b 16/2/128 heads, 64 rows over up to 64 keys",
				"prefill attention, phi-3.5-mini 32/32/96 heads, 64 rows over up to 64 keys",
				"prefill attention, mistral-7b 32/8/128 heads, 64 rows over up to 64 keys",
				"tiled K-quant GEMM, mistral-7b ffn_gate and ffn_up 14336x4096 Q4_K, 64 rows",
				"tiled K-quant GEMM, mistral-7b ffn_down 4096x14336 Q6_K, 64 rows",
				"LM head GEMV, qwen2.5-3b output 151936x2048 Q6_K, 1 row");
		assertThat(t).allSatisfy(x -> {
			assertThat(x.millis()).as(x.name()).hasSize(3).allSatisfy(ms -> assertThat(ms).isPositive().isFinite());
			assertThat(x.medianMillis()).as(x.name()).isLessThanOrEqualTo(x.maxMillis());
		});
	}

	/**
	 * Guards against timing the wrong thing (events on another stream than the launch read near zero whatever
	 * the work): sixteen times the rows over sixteen times the keys is about 256 times the attention work.
	 */
	@Test
	@Tag("gpu")
	@DisplayName("a wider window takes longer: the attention reading follows the work, not launch overhead")
	void attentionReadingGrowsWithTheWindow() {
		assumeTrue(CudaAvailability.isAvailable(), "no CUDA device");
		double narrow = attentionMedian(GpuKernelTimings.measure(64, 3));
		double wide = attentionMedian(GpuKernelTimings.measure(1024, 3));
		assertThat(wide).as("1024-row median over 64-row median").isGreaterThan(8 * narrow);
	}

	@Test
	@DisplayName("without CUDA, refuses with the reason")
	void refusesWithoutCuda() {
		assumeFalse(CudaAvailability.isAvailable(), "CUDA is present");
		assertThatThrownBy(() -> GpuKernelTimings.measure(64, 1)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("CUDA");
	}

	private static double attentionMedian(List<GpuKernelTimings.Timing> timings) {
		return timings.stream().filter(x -> x.name().startsWith("prefill attention, mistral-7b")).findFirst()
				.orElseThrow().medianMillis();
	}
}
