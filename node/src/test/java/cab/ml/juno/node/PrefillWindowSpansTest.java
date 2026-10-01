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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

import cab.ml.juno.registry.ShardAssignment;

/**
 * A prefill window is covered by spans end to end, so its time can be broken down per term
 * with nothing left unnamed: the per-op spans ({@code juno.RmsNorm}, {@code Rope},
 * {@code Attention}, {@code ResidualAdd}, {@code SwiGlu}) on every handler with a window
 * path, and {@code juno.WindowStep} for the rest (embedding, each projection call with its
 * copy-out, the KV write, the LM head). Synthetic models on the CPU backend; the spans are
 * the same whichever backend runs the matmuls.
 */
@DisplayName("Prefill window spans")
class PrefillWindowSpansTest {

	private static final List<String> EVENTS = List.of("juno.RmsNorm", "juno.Rope", "juno.Attention",
			"juno.ResidualAdd", "juno.SwiGlu", "juno.WindowStep");
	private static final int W = 8;
	private static final int LAYERS = 2;

	@Test
	@DisplayName("LLaMA family: projections, KV write, embedding and LM head are spanned")
	void llamaWindow_spansTheWorkOutsideThePerOpEvents(@TempDir Path dir) throws Exception {
		int vocab = 64, h = 16, heads = 2;
		LlamaTransformerHandler handler = LlamaTransformerHandler.newTestInstance(vocab, h, heads, heads, LAYERS, 0,
				LAYERS, true, true, null);
		ShardContext ctx = ShardContext.from(new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, true), vocab,
				h, heads);

		Map<String, List<RecordedEvent>> ev = record(dir.resolve("llama.jfr"),
				() -> handler.forwardBatch(BatchForwardRequest.withTokens("r1", tokens(), 0), ctx));

		Map<String, List<RecordedEvent>> steps = byStep(ev.get("juno.WindowStep"));
		assertThat(steps.get("embed")).hasSize(1);
		assertThat(steps.get("projection")).as("q, k, v, o, gate, up, down per layer").hasSize(7 * LAYERS);
		assertThat(steps.get("kv_write")).hasSize(LAYERS);
		assertThat(steps.get("lm_head")).hasSize(1);
		assertWindow(ev.get("juno.WindowStep"));
	}

	@Test
	@DisplayName("Phi-3: the window path emits the per-op spans and the window steps")
	void phi3Window_emitsPerOpSpansAndWindowSteps(@TempDir Path dir) throws Exception {
		int h = 256, heads = 8, inter = 256, vocab = 256;
		Path gguf = Phi3TransformerHandlerTest.buildSyntheticPhiGguf(dir, h, heads, heads, inter, LAYERS, vocab);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, vocab, h, heads);
		Phi3TransformerHandler handler = Phi3TransformerHandler.load(gguf, ctx);

		Map<String, List<RecordedEvent>> ev = record(dir.resolve("phi3.jfr"),
				() -> handler.forwardBatch(BatchForwardRequest.withTokens("r1", tokens(), 0), ctx));

		assertThat(ev.get("juno.RmsNorm")).hasSize(2 * LAYERS);
		assertThat(ev.get("juno.Rope")).hasSize(LAYERS);
		assertThat(ev.get("juno.Attention")).hasSize(LAYERS);
		assertThat(ev.get("juno.ResidualAdd")).hasSize(2 * LAYERS);
		assertThat(ev.get("juno.SwiGlu")).hasSize(LAYERS);
		Map<String, List<RecordedEvent>> steps = byStep(ev.get("juno.WindowStep"));
		assertThat(steps.get("embed")).hasSize(1);
		assertThat(steps.get("projection")).as("fused qkv, o, fused gate-up, down per layer").hasSize(4 * LAYERS);
		assertThat(steps.get("kv_write")).hasSize(LAYERS);
		assertThat(steps.get("lm_head")).hasSize(1);
		for (String name : EVENTS)
			assertWindow(ev.get(name));
	}

	@Test
	@DisplayName("Qwen3: the window path emits the per-op spans, the per-head Q/K norm included")
	void qwen3Window_emitsPerOpSpansAndWindowSteps(@TempDir Path dir) throws Exception {
		int h = 256, heads = 8, vocab = 256;
		Path gguf = Qwen3AttentionNormTest.buildSyntheticQwen3GgufWithKeyLength(dir, h, heads, 4, 32, LAYERS);
		ShardContext ctx = new ShardContext("n1", 0, LAYERS, true, true, vocab, h, heads);
		Qwen3TransformerHandler handler = Qwen3TransformerHandler.load(gguf, ctx);

		Map<String, List<RecordedEvent>> ev = record(dir.resolve("qwen3.jfr"),
				() -> handler.forwardBatch(BatchForwardRequest.withTokens("r1", tokens(), 0), ctx));

		assertThat(ev.get("juno.RmsNorm")).as("attention, per-head Q/K and FFN norms").hasSize(3 * LAYERS);
		assertThat(ev.get("juno.Rope")).hasSize(LAYERS);
		assertThat(ev.get("juno.Attention")).hasSize(LAYERS);
		assertThat(ev.get("juno.ResidualAdd")).hasSize(2 * LAYERS);
		assertThat(ev.get("juno.SwiGlu")).hasSize(LAYERS);
		Map<String, List<RecordedEvent>> steps = byStep(ev.get("juno.WindowStep"));
		assertThat(steps.get("embed")).hasSize(1);
		assertThat(steps.get("projection")).hasSize(7 * LAYERS);
		assertThat(steps.get("kv_write")).hasSize(LAYERS);
		assertThat(steps.get("lm_head")).hasSize(1);
		for (String name : EVENTS)
			assertWindow(ev.get(name));
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private static int[] tokens() {
		int[] t = new int[W];
		for (int i = 0; i < W; i++)
			t[i] = i + 1;
		return t;
	}

	private static void assertWindow(List<RecordedEvent> events) {
		assertThat(events).isNotEmpty();
		for (RecordedEvent e : events) {
			assertThat(e.getInt("windowSize")).as(e.getEventType().getName()).isEqualTo(W);
			assertThat(e.getInt("startPosition")).as(e.getEventType().getName()).isEqualTo(0);
		}
	}

	private static Map<String, List<RecordedEvent>> byStep(List<RecordedEvent> events) {
		Map<String, List<RecordedEvent>> m = new HashMap<>();
		for (String s : List.of("embed", "projection", "bias_add", "kv_write", "lm_head"))
			m.put(s, new ArrayList<>());
		for (RecordedEvent e : events)
			m.computeIfAbsent(e.getString("step"), k -> new ArrayList<>()).add(e);
		return m;
	}

	private static Map<String, List<RecordedEvent>> record(Path jfr, Runnable body) throws Exception {
		try (Recording rec = new Recording()) {
			for (String name : EVENTS)
				rec.enable(name);
			rec.start();
			body.run();
			rec.stop();
			rec.dump(jfr);
		}
		Map<String, List<RecordedEvent>> byName = new HashMap<>();
		for (String name : EVENTS)
			byName.put(name, new ArrayList<>());
		try (RecordingFile rf = new RecordingFile(jfr)) {
			while (rf.hasMoreEvents()) {
				RecordedEvent e = rf.readEvent();
				List<RecordedEvent> l = byName.get(e.getEventType().getName());
				if (l != null)
					l.add(e);
			}
		}
		return byName;
	}
}
