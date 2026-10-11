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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

/**
 * Steady-state CPU decode allocates less than half of what it did before the
 * decode scratch: on Mistral 7B, at most {@link #CEILING_BYTES_PER_TOKEN} bytes per
 * generated token, counted over every thread
 * ({@code com.sun.management.ThreadMXBean#getTotalThreadAllocatedBytes}, the
 * counters JFR's {@code jdk.ThreadAllocationStatistics} reads).
 *
 * <p>The ceiling is half the 20.8 MB per token the pre-fix decode allocated on this
 * model, the CPU tier's allocation threshold. The first tokens warm the JIT and
 * grow the KV cache's first pages, so they are not counted.
 */
class CpuDecodeAllocationLiveTest {

	static final long CEILING_BYTES_PER_TOKEN = 10_400_000L;
	private static final int WARMUP = 16;
	private static final int MEASURED = 32;

	@Test
	void mistralDecodeAllocatesUnderTheCeiling() throws Exception {
		Path model = ContextShiftLiveTest.model("mistral-7b-instruct-v0.1-q4_k_m.gguf");
		assumeTrue(model.toFile().exists(), "Mistral 7B not present");
		com.sun.management.ThreadMXBean threads = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		assumeTrue(threads.isThreadAllocatedMemorySupported(), "thread allocation counters not supported");
		ShardContext ctx;
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig c = LlamaConfig.from(r);
			ctx = new ShardContext("n0", 0, c.numLayers(), true, true, c.vocabSize(), c.hiddenDim(), c.numHeads());
		}
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, ctx, CpuMatVec.INSTANCE);
		try {
			int pos = 0;
			for (; pos < WARMUP; pos++)
				h.forward(ForwardRequest.withTokens("alloc", new int[] { ContextShiftLiveCheck.token(pos) }, pos), ctx);
			long before = threads.getTotalThreadAllocatedBytes();
			for (int i = 0; i < MEASURED; i++, pos++)
				h.forward(ForwardRequest.withTokens("alloc", new int[] { ContextShiftLiveCheck.token(pos) }, pos), ctx);
			long perToken = (threads.getTotalThreadAllocatedBytes() - before) / MEASURED;
			System.out.printf("CPU decode allocation, Mistral 7B: %.2f MB per token (ceiling %.1f MB)%n", perToken / 1e6,
					CEILING_BYTES_PER_TOKEN / 1e6);
			assertThat(perToken).as("bytes allocated per decoded token").isLessThanOrEqualTo(CEILING_BYTES_PER_TOKEN);
		} finally {
			h.evict("alloc");
		}
	}
}
