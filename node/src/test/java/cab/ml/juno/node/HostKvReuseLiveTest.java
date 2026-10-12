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
import java.util.function.Function;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.SessionKvTensor;

/**
 * On a real model, a request's prefill window takes the host KV tensors an evicted
 * request left instead of allocating its own, and computes the same logits as on
 * new tensors, bit for bit. One case per handler that reuses them: the LLaMA family,
 * Phi-3 and Qwen3. CPU backend, short windows.
 */
@DisplayName("Host KV tensors reused across requests on a real model")
class HostKvReuseLiveTest {

	private static final Path MODELS = Path.of(System.getProperty("user.dir")).endsWith("node")
			? Path.of(System.getProperty("user.dir")).getParent().resolve("models")
			: Path.of("models");

	private static final int SHORT = 40;
	private static final int LONG = 100;

	@Test
	@DisplayName("LLaMA family (TinyLlama)")
	void llama() throws Exception {
		Path model = MODELS.resolve("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
		assumeTrue(model.toFile().exists(), "model not present");
		ShardContext shard = shard(model);
		LlamaTransformerHandler h = LlamaTransformerHandler.load(model, shard);
		check(h, shard, h::hostKv);
	}

	@Test
	@DisplayName("Phi-3 (Phi-3.5-mini)")
	void phi3() throws Exception {
		Path model = MODELS.resolve("Phi-3.5-mini-instruct-Q4_K_M.gguf");
		assumeTrue(model.toFile().exists(), "model not present");
		ShardContext shard = shard(model);
		Phi3TransformerHandler h = Phi3TransformerHandler.load(model, shard);
		check(h, shard, h::hostKv);
	}

	@Test
	@DisplayName("Qwen3 (Qwen3-1.7B)")
	void qwen3() throws Exception {
		Path model = MODELS.resolve("Qwen3-1.7B-Q4_K_M.gguf");
		assumeTrue(model.toFile().exists(), "model not present");
		ShardContext shard = shard(model);
		Qwen3TransformerHandler h = Qwen3TransformerHandler.load(model, shard);
		check(h, shard, h::hostKv);
	}

	/**
	 * A long request stays open while a short one prefills on new tensors (the
	 * reference); both are evicted, and the same short prompt prefills again. It must
	 * take the long request's tensors (the larger set is the one kept), allocate at
	 * least their worth of bytes less than the reference did, and give its logits.
	 */
	private static void check(ForwardPassHandler h, ShardContext shard, Function<String, SessionKvTensor[][]> hostKv) {
		int[] shortPrompt = tokens(SHORT, 11);
		int[] longPrompt = tokens(LONG, 29);
		h.forwardBatch(BatchForwardRequest.withTokens("long", longPrompt, 0), shard);
		SessionKvTensor[][] longKv = hostKv.apply("long");

		long before = allocated();
		float[] reference = h.forwardBatch(BatchForwardRequest.withTokens("ref", shortPrompt, 0), shard).lastLogits();
		long referenceBytes = allocated() - before;
		SessionKvTensor[][] refKv = hostKv.apply("ref");
		long refKvBytes = 0;
		for (SessionKvTensor[] side : refKv)
			for (SessionKvTensor t : side)
				refKvBytes += (long) t.capacityTokens() * t.kvDim() * Float.BYTES;
		h.evict("ref");
		h.evict("long");

		before = allocated();
		float[] again = h.forwardBatch(BatchForwardRequest.withTokens("again", shortPrompt, 0), shard).lastLogits();
		long againBytes = allocated() - before;
		SessionKvTensor[][] againKv = hostKv.apply("again");
		h.evict("again");

		for (int side = 0; side < 2; side++)
			for (int li = 0; li < longKv[side].length; li++)
				assertThat(againKv[side][li]).as("side %d layer %d", side, li).isSameAs(longKv[side][li]);
		assertThat(again).as("logits on reused tensors").containsExactly(reference);
		System.out.printf(java.util.Locale.ROOT,
				"HOST-KV-REUSE %s: window allocation %.1f MB on new tensors, %.1f MB reused (KV %.1f MB)%n",
				h.getClass().getSimpleName(), referenceBytes / 1e6, againBytes / 1e6, refKvBytes / 1e6);
		assertThat(referenceBytes - againBytes).as("bytes the reused window did not allocate")
				.isGreaterThanOrEqualTo(refKvBytes * 9 / 10);
	}

	/** Bytes the calling thread has allocated so far; the KV tensors are allocated on it. */
	private static long allocated() {
		return ((com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean()).getCurrentThreadAllocatedBytes();
	}

	private static int[] tokens(int n, int seed) {
		int[] ids = new int[n];
		for (int i = 0; i < n; i++)
			ids[i] = 100 + (i * seed) % 900;
		return ids;
	}

	private static ShardContext shard(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = LlamaConfig.from(r);
			return new ShardContext("n0", 0, cfg.numLayers(), true, true, cfg.vocabSize(), cfg.hiddenDim(),
					cfg.numHeads());
		}
	}
}
