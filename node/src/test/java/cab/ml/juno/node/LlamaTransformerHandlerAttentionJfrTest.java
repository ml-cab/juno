package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

import cab.ml.juno.registry.ShardAssignment;

/**
 * Verifies {@link AttentionEvent} ("juno.Attention") fires with correct
 * {@code windowSize}/{@code startPosition}/{@code contextLength} for both the
 * batched-prefill loop ({@link LlamaTransformerHandler#transformerLayerBatch})
 * and single-token decode ({@link LlamaTransformerHandler#transformerLayer}).
 *
 * <p>Added to give the {@code O(seq^2)} attention cost at long prefill windows
 * its own JFR span, separate from {@code juno.MatVec} — see
 * {@code docs/perf-compare/README.md} "GPU batched-prefill GEMM bake-off —
 * Tier 17" long-window caveat.
 */
@DisplayName("LlamaTransformerHandler — juno.Attention JFR span")
class LlamaTransformerHandlerAttentionJfrTest {

	private static final String ATTENTION_EVENT = "juno.Attention";
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

	private List<RecordedEvent> record(Path jfrFile, Runnable body) throws Exception {
		try (Recording rec = new Recording()) {
			rec.enable(ATTENTION_EVENT);
			rec.start();
			body.run();
			rec.stop();
			rec.dump(jfrFile);
		}
		List<RecordedEvent> events = new ArrayList<>();
		try (RecordingFile rf = new RecordingFile(jfrFile)) {
			while (rf.hasMoreEvents()) {
				RecordedEvent ev = rf.readEvent();
				if (ATTENTION_EVENT.equals(ev.getEventType().getName())) {
					events.add(ev);
				}
			}
		}
		return events;
	}

	@Test
	@DisplayName("batched prefill emits one event per layer with windowSize = prompt length")
	void batchedPrefill_emitsOneEventPerLayer_withFullWindowSize(@TempDir Path dir) throws Exception {
		int[] prompt = { 1, 2, 3, 4, 5, 6, 7, 8 };
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();

		List<RecordedEvent> events = record(dir.resolve("prefill.jfr"),
				() -> handler.forwardBatch(BatchForwardRequest.withTokens("r1", prompt, 0), ctx));

		assertThat(events).as("one juno.Attention event per layer").hasSize(LAYERS);
		for (RecordedEvent ev : events) {
			assertThat(ev.getInt("windowSize")).isEqualTo(prompt.length);
			assertThat(ev.getInt("startPosition")).isEqualTo(0);
			assertThat(ev.getInt("contextLength")).isEqualTo(prompt.length);
		}
	}

	@Test
	@DisplayName("chunked prefill windows report the correct startPosition/contextLength per chunk")
	void chunkedPrefill_reportsStartPositionPerChunk(@TempDir Path dir) throws Exception {
		int[] chunk1 = { 1, 2, 3, 4 };
		int[] chunk2 = { 5, 6, 7, 8 };
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();

		List<RecordedEvent> events = record(dir.resolve("chunked.jfr"), () -> {
			handler.forwardBatch(BatchForwardRequest.withTokens("r2", chunk1, 0), ctx);
			handler.forwardBatch(BatchForwardRequest.withTokens("r2", chunk2, chunk1.length), ctx);
		});

		assertThat(events).hasSize(LAYERS * 2);
		long secondChunkEvents = events.stream()
				.filter(ev -> ev.getInt("startPosition") == chunk1.length)
				.count();
		assertThat(secondChunkEvents).isEqualTo(LAYERS);
		events.stream()
				.filter(ev -> ev.getInt("startPosition") == chunk1.length)
				.forEach(ev -> {
					assertThat(ev.getInt("windowSize")).isEqualTo(chunk2.length);
					assertThat(ev.getInt("contextLength")).isEqualTo(chunk1.length + chunk2.length);
				});
	}

	@Test
	@DisplayName("single-token decode emits windowSize = 1 with growing contextLength")
	void decode_emitsWindowSizeOne_perToken(@TempDir Path dir) throws Exception {
		LlamaTransformerHandler handler = handler();
		ShardContext ctx = ctx();
		// Prime the KV cache at position 0 (sequential-fallback prefill of one token).
		handler.forward(ForwardRequest.withTokens("r3", new int[] { 1 }, 0), ctx);

		List<RecordedEvent> events = record(dir.resolve("decode.jfr"),
				() -> handler.forward(ForwardRequest.withTokens("r3", new int[] { 2 }, 1), ctx));

		assertThat(events).hasSize(LAYERS);
		for (RecordedEvent ev : events) {
			assertThat(ev.getInt("windowSize")).isEqualTo(1);
			assertThat(ev.getInt("startPosition")).isEqualTo(1);
			assertThat(ev.getInt("contextLength")).isEqualTo(2);
		}
	}
}
