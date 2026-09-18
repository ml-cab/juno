package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.ArrayList;
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
 * Verifies the Tier 19 baseline-measurement JFR spans ({@code juno.RmsNorm},
 * {@code juno.Rope}, {@code juno.ResidualAdd}, {@code juno.SwiGlu}) fire with
 * correct {@code windowSize}/{@code startPosition}/{@code dimension} for both
 * the batched-prefill loop ({@link LlamaTransformerHandler#transformerLayerBatch})
 * and single-token decode ({@link LlamaTransformerHandler#transformerLayer}).
 *
 * <p>These events are measurement only — no behavior change — added so the
 * scalar CPU cost of these four ops (believed, per the 2026-09-18 Nsight pass
 * documented in {@code docs/infra-plan/PLAN-Infra-PERF-ANALYSIS.md}, to be part
 * of the per-launch host/FFI overhead that leaves the GPU idle 73-82% of decode
 * wall time) is directly visible before any GPU-kernel work moves them off the
 * CPU. See {@code docs/infra-plan/PLAN-Infra-Tier19.md}.
 */
@DisplayName("LlamaTransformerHandler — elementwise-op JFR spans")
class LlamaTransformerHandlerElementwiseOpsJfrTest {

	private static final List<String> EVENT_NAMES =
			List.of("juno.RmsNorm", "juno.Rope", "juno.ResidualAdd", "juno.SwiGlu");
	private static final int VOCAB = 64;
	private static final int H = 16;
	private static final int NH = 2;
	private static final int KVH = 2;
	private static final int LAYERS = 2;

	private LlamaTransformerHandler handler() {
		return LlamaTransformerHandler.newTestInstance(VOCAB, H, NH, KVH, LAYERS, 0, LAYERS, true, true, null);
	}

	private ShardContext ctx() {
		ShardAssignment a = new ShardAssignment("n1", "localhost", 0, 0, LAYERS, true, true);
		return ShardContext.from(a, VOCAB, H, NH);
	}

	private Map<String, List<RecordedEvent>> record(Path jfrFile, Runnable body) throws Exception {
		try (Recording rec = new Recording()) {
			for (String name : EVENT_NAMES)
				rec.enable(name);
			rec.start();
			body.run();
			rec.stop();
			rec.dump(jfrFile);
		}
		Map<String, List<RecordedEvent>> byName = new java.util.HashMap<>();
		for (String name : EVENT_NAMES)
			byName.put(name, new ArrayList<>());
		try (RecordingFile rf = new RecordingFile(jfrFile)) {
			while (rf.hasMoreEvents()) {
				RecordedEvent ev = rf.readEvent();
				String type = ev.getEventType().getName();
				if (byName.containsKey(type))
					byName.get(type).add(ev);
			}
		}
		return byName;
	}

	@Test
	@DisplayName("batched prefill emits two RmsNorm/ResidualAdd events and one Rope/SwiGlu event per layer")
	void batchedPrefill_emitsExpectedCountsAndFields(@TempDir Path dir) throws Exception {
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8 };
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();

		Map<String, List<RecordedEvent>> events = record(dir.resolve("prefill.jfr"),
				() -> handler.forwardBatch(BatchForwardRequest.withTokens("r1", prompt, 0), ctx));

		// One RmsNorm/ResidualAdd call site for attention, one for FFN, per layer.
		assertThat(events.get("juno.RmsNorm")).as("attn-norm + ffn-norm per layer").hasSize(LAYERS * 2);
		assertThat(events.get("juno.ResidualAdd")).as("attn residual + ffn residual per layer").hasSize(LAYERS * 2);
		// Rope rotates Q and K in one call site; SwiGLU runs once per layer.
		assertThat(events.get("juno.Rope")).as("one rope call per layer").hasSize(LAYERS);
		assertThat(events.get("juno.SwiGlu")).as("one swiglu call per layer").hasSize(LAYERS);

		for (RecordedEvent ev : events.get("juno.RmsNorm")) {
			assertThat(ev.getInt("windowSize")).isEqualTo(prompt.length);
			assertThat(ev.getInt("startPosition")).isEqualTo(0);
			assertThat(ev.getInt("dimension")).isEqualTo(H);
		}
		for (RecordedEvent ev : events.get("juno.Rope")) {
			assertThat(ev.getInt("windowSize")).isEqualTo(prompt.length);
			assertThat(ev.getInt("startPosition")).isEqualTo(0);
			assertThat(ev.getInt("dimension")).isEqualTo(NH * (H / NH) + KVH * (H / NH));
		}
		for (RecordedEvent ev : events.get("juno.ResidualAdd")) {
			assertThat(ev.getInt("windowSize")).isEqualTo(prompt.length);
			assertThat(ev.getInt("startPosition")).isEqualTo(0);
			assertThat(ev.getInt("dimension")).isEqualTo(H);
		}
	}

	@Test
	@DisplayName("single-token decode emits windowSize = 1 for every op")
	void decode_emitsWindowSizeOne_perToken(@TempDir Path dir) throws Exception {
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();
		// Prime the KV cache at position 0 (sequential-fallback prefill of one token).
		handler.forward(ForwardRequest.withTokens("r3", new int[] { 1 }, 0), ctx);

		Map<String, List<RecordedEvent>> events = record(dir.resolve("decode.jfr"),
				() -> handler.forward(ForwardRequest.withTokens("r3", new int[] { 2 }, 1), ctx));

		assertThat(events.get("juno.RmsNorm")).hasSize(LAYERS * 2);
		assertThat(events.get("juno.ResidualAdd")).hasSize(LAYERS * 2);
		assertThat(events.get("juno.Rope")).hasSize(LAYERS);
		assertThat(events.get("juno.SwiGlu")).hasSize(LAYERS);

		for (String name : EVENT_NAMES) {
			for (RecordedEvent ev : events.get(name)) {
				assertThat(ev.getInt("windowSize")).isEqualTo(1);
				assertThat(ev.getInt("startPosition")).isEqualTo(1);
			}
		}
	}
}
