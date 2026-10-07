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

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Qwen3-1.7B's prefill window normalizes each Q and K head, rotates and attends
 * inside the prefill-window device region, so per layer only the K and V rows for
 * the host KV tensors cross the bus.
 *
 * <p>Read from a JFR recording of one 512-token window: no host per-head norm, RoPE
 * or attention, no Q download, no attention-output upload, no copy of the attention
 * kernel's host-fed path, and the window's host-device bytes at most what the K and
 * V rows and the residual (once each way) move, 125.8 MB for this model's 28 layers
 * (2 x 512 x 1024 x 4 B x 28 + 2 x 512 x 2048 x 4 B), plus 4.2 MB for the attention
 * tables (about 0.4 MB) and the LM head's input row and logits (about 0.6 MB): 130 MB. Bit identity against the region-off path is
 * {@link PrefillRegionHandlerParityTest}'s.
 */
@Tag("gpu")
@DisplayName("Qwen3 prefill window - per-head norm, RoPE and attention inside the device region")
class Qwen3PrefillRegionAttentionTest {

	private static final String MODEL = "Qwen3-1.7B-Q4_K_M.gguf";
	private static final int WINDOW = 512;
	/** K and V rows of every layer, the residual once each way, and the attention tables. */
	private static final long MAX_WINDOW_BYTES = 130_000_000L;

	private static GpuContext gpu;

	@TempDir
	Path tmp;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		gpu = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (gpu != null)
			gpu.close();
	}

	@Test
	@DisplayName("a 512-token window keeps the head norms, RoPE and attention on the device")
	void windowAttendsInsideTheRegion() throws Exception {
		Path model = model();
		assumeTrue(model.toFile().exists(), "Skipping - model not found: " + model);
		ShardContext shard = wholeModel(model);
		CudaMatVec backend = new CudaMatVec(gpu);
		ForwardPassHandler h = ForwardPassHandlerLoader.load(model, shard, backend);
		List<RecordedEvent> events;
		try {
			assertThat(h.prefillRegionActive()).as("the default must build the region on CUDA").isTrue();
			int[] tokens = new int[WINDOW];
			for (int i = 0; i < WINDOW; i++)
				tokens[i] = 1000 + (int) ((300L + 7919L * i) % (shard.vocabSize() / 2));
			// One unrecorded window first, so the recorded one sees grown buffers and mirrors.
			h.forwardBatch(BatchForwardRequest.withTokens("warm", tokens, 0), shard);
			h.evict("warm");
			Path jfr = tmp.resolve("qwen3-region.jfr");
			try (Recording rec = new Recording()) {
				rec.enable("juno.DeviceStaging").withThreshold(Duration.ZERO);
				rec.enable("juno.RmsNorm").withThreshold(Duration.ZERO);
				rec.enable("juno.Rope").withThreshold(Duration.ZERO);
				rec.enable("juno.Attention").withThreshold(Duration.ZERO);
				rec.setDestination(jfr);
				rec.start();
				h.forwardBatch(BatchForwardRequest.withTokens("a", tokens, 0), shard);
				rec.stop();
			}
			h.evict("a");
			events = RecordingFile.readAllEvents(jfr);
		} finally {
			h.releaseGpuResources();
			backend.releaseScratch();
		}

		long h2d = bytes(events, "H2D");
		long d2h = bytes(events, "D2H");
		System.out.printf(Locale.ROOT, "QWEN3-PREFILL-REGION window=%d H2D=%.1f MB D2H=%.1f MB total=%.1f MB%n", WINDOW,
				h2d / 1e6, d2h / 1e6, (h2d + d2h) / 1e6);
		assertThat(count(events, "juno.RmsNorm")).as("host norms (the per-head Q/K norm) in a region window").isZero();
		assertThat(count(events, "juno.Rope")).as("host RoPE events in a region window").isZero();
		assertThat(count(events, "juno.Attention")).as("host attention events in a region window").isZero();
		assertThat(siteCopies(events, "D2H", "materialize(prefill q)")).as("Q rows downloaded in a region window").isZero();
		assertThat(siteCopies(events, "H2D", "upload(prefill attention)")).as("attention output uploaded in a region window")
				.isZero();
		assertThat(siteCopies(events, "H2D", "memcpy(gqa qBatch H2D)")).as("queries fed to the host-side attention call")
				.isZero();
		assertThat(h2d + d2h).as("host-device bytes of one 512-token window").isLessThanOrEqualTo(MAX_WINDOW_BYTES);
	}

	private static long bytes(List<RecordedEvent> events, String direction) {
		return staging(events).stream().filter(e -> direction.equals(e.getString("direction")))
				.mapToLong(e -> e.getLong("bytes")).sum();
	}

	private static long siteCopies(List<RecordedEvent> events, String direction, String site) {
		return staging(events).stream().filter(e -> direction.equals(e.getString("direction")))
				.filter(e -> site.equals(e.getString("site")))
				.mapToLong(e -> e.getLong("copies")).sum();
	}

	private static List<RecordedEvent> staging(List<RecordedEvent> events) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging")).toList();
	}

	private static long count(List<RecordedEvent> events, String name) {
		return events.stream().filter(e -> e.getEventType().getName().equals(name)).count();
	}

	private static ShardContext wholeModel(Path model) throws Exception {
		try (GgufReader r = GgufReader.open(model)) {
			LlamaConfig cfg = Qwen3Config.from(r).base();
			return new ShardContext("qwen3-prefill-region", 0, cfg.numLayers(), true, true, cfg.vocabSize(),
					cfg.hiddenDim(), cfg.numHeads());
		}
	}

	private static Path model() {
		Path here = Path.of(System.getProperty("user.dir"));
		Path root = here.endsWith("node") ? here.getParent() : here;
		return root.resolve("models").resolve(MODEL);
	}
}
