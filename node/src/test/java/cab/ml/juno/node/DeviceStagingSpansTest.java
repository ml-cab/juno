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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * {@code juno.DeviceStaging} and {@code juno.WeightDequant} on a real CUDA device: every
 * host-device copy is totalled under its site and phase with the bytes it moved, prefill-width
 * copies with a duration measured on the device, the dequantization of packed weights apart
 * from the copies, and recording changes no output.
 *
 * <p>The byte counts are the contract a later residency change is scored against (bytes staged
 * per prefill window), so they are asserted exactly rather than as "positive".
 */
@Tag("gpu")
@DisplayName("juno.DeviceStaging / juno.WeightDequant on CUDA")
class DeviceStagingSpansTest {

	private static final int ROWS = 256;
	private static final int COLS = 512;
	/** Q4_K: 144 bytes per 256-element super-block. */
	private static final int Q4K_BLOCK_BYTES = 144;

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@TempDir
	Path tmp;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		mv = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a prefill-width FP16 GEMM counts one H2D and one D2H with exact bytes, timed on the device")
	void prefillGemm_countsBothCopies() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 1), ROWS, COLS);
		float[][] x = randomRows(32, COLS, 2);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		RecordedEvent h2d = cell(events, "cudaMemcpyAsync(xh H2D batched-gemm)", "prefill");
		RecordedEvent d2h = cell(events, "cudaMemcpyAsync(y D2H batched-gemm)", "prefill");
		assertThat(h2d.getString("direction")).isEqualTo("H2D");
		assertThat(h2d.getLong("bytes")).isEqualTo((long) COLS * 32 * Short.BYTES);
		assertThat(d2h.getLong("bytes")).isEqualTo((long) ROWS * 32 * Float.BYTES);
		for (RecordedEvent e : List.of(h2d, d2h)) {
			assertThat(e.getLong("copies")).isEqualTo(1);
			assertThat(e.getLong("timedCopies")).isEqualTo(1);
			assertThat(e.getLong("transferNanos")).isPositive();
		}
	}

	@Test
	@DisplayName("a single-token GEMV counts decode-width copies with their bytes, untimed")
	void decodeGemv_isCountedNotTimed() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 3), ROWS, COLS);
		float[] x = randomFloats(COLS, 4);
		List<RecordedEvent> events = record(() -> {
			mv.sgemv(a, x);
			mv.sgemv(a, x);
		});
		a.close();

		RecordedEvent d2h = cell(events, "cudaMemcpyAsync(y D2H)", "decode");
		assertThat(d2h.getLong("copies")).isEqualTo(2);
		assertThat(d2h.getLong("bytes")).isEqualTo(2L * ROWS * Float.BYTES);
		assertThat(d2h.getLong("timedCopies")).isZero();
		assertThat(cell(events, "cudaMemcpyAsync(xh H2D)", "decode").getLong("bytes"))
				.isEqualTo(2L * COLS * Short.BYTES);
	}

	@Test
	@DisplayName("a batched Q4_K GEMM counts the device dequantization apart from the copies")
	void q4kGemm_countsDeviceDequant() throws Exception {
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 5), ROWS, COLS);
		float[][] x = randomRows(16, COLS, 6);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		List<RecordedEvent> dq = named(events, "juno.WeightDequant");
		assertThat(dq).hasSize(1);
		assertThat(dq.get(0).getString("format")).isEqualTo("Q4_K");
		assertThat(dq.get(0).getString("timing")).isEqualTo("device");
		assertThat(dq.get(0).getLong("count")).isEqualTo(1);
		assertThat(dq.get(0).getLong("timedCount")).isEqualTo(1);
		assertThat(dq.get(0).getLong("dequantNanos")).isPositive();
		assertThat(cell(events, "cudaMemcpyAsync(xh H2D q4k-batched-gemm)", "prefill").getLong("bytes"))
				.isEqualTo((long) COLS * 16 * Short.BYTES);
	}

	@Test
	@DisplayName("a weight upload is an other-phase copy timed on the host")
	void weightUpload_isOtherPhaseHostTimed() throws Exception {
		float[] w = randomFloats(ROWS * COLS, 7);
		DeviceHalfMatrix[] holder = new DeviceHalfMatrix[1];
		List<RecordedEvent> events = record(() -> holder[0] = mv.uploadHalf(w, ROWS, COLS));
		holder[0].close();

		RecordedEvent up = cell(events, "memcpy(A FP16 H2D)", "other");
		assertThat(up.getLong("bytes")).isEqualTo((long) ROWS * COLS * Short.BYTES);
		assertThat(up.getLong("timedCopies")).isEqualTo(1);
		assertThat(up.getLong("transferNanos")).isPositive();
	}

	@Test
	@DisplayName("resident activations: a multi-row upload and materialize are timed once the chain synchronizes")
	void residentChain_countsUploadAndMaterialize() throws Exception {
		float[][] x = randomRows(5, 256, 8);
		float[][] out = new float[5][];
		List<RecordedEvent> events = record(() -> {
			try (ResidentChain chain = ResidentChain.open(ctx)) {
				ResidentActivation act = chain.allocate(8, 256);
				act.upload(x);
				act.materialize(out);
			}
		});
		RecordedEvent up = cell(events, "upload(resident activation)", "prefill");
		RecordedEvent down = cell(events, "materialize(resident activation)", "prefill");
		for (RecordedEvent e : List.of(up, down)) {
			assertThat(e.getLong("bytes")).isEqualTo(5L * 256 * Float.BYTES);
			assertThat(e.getLong("timedCopies")).isEqualTo(1);
		}
	}

	@Test
	@DisplayName("recording changes no output: the same GEMM is bit-identical with and without it")
	void recording_isBitIdentical() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 9), ROWS, COLS);
		float[][] x = randomRows(32, COLS, 10);
		float[][] plain = mv.sgemm(a, x);
		float[][][] recorded = new float[1][][];
		record(() -> recorded[0] = mv.sgemm(a, x));
		a.close();
		for (int b = 0; b < x.length; b++)
			assertThat(recorded[0][b]).as("row " + b).containsExactly(plain[b]);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("spans-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceStaging").withThreshold(java.time.Duration.ZERO);
			rec.enable("juno.WeightDequant").withThreshold(java.time.Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		assertThat(Files.size(jfr)).isPositive();
		return RecordingFile.readAllEvents(jfr);
	}

	private static List<RecordedEvent> named(List<RecordedEvent> events, String name) {
		return events.stream().filter(e -> e.getEventType().getName().equals(name)).toList();
	}

	/** The one total for {@code site} in {@code phase}. */
	private static RecordedEvent cell(List<RecordedEvent> events, String site, String phase) {
		List<RecordedEvent> matching = named(events, "juno.DeviceStaging").stream()
				.filter(e -> site.equals(e.getString("site")) && phase.equals(e.getString("phase"))).toList();
		assertThat(matching).as(site + " / " + phase).hasSize(1);
		return matching.get(0);
	}

	private static float[] randomFloats(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = r.nextFloat() * 2f - 1f;
		return v;
	}

	private static float[][] randomRows(int rows, int cols, long seed) {
		float[][] x = new float[rows][];
		for (int i = 0; i < rows; i++)
			x[i] = randomFloats(cols, seed * 1000 + i);
		return x;
	}

	/** Random Q4_K super-blocks with small finite FP16 scales, so the dequantized weights stay finite. */
	private static byte[] randomQ4K(int rows, int cols, long seed) {
		Random r = new Random(seed);
		int blocks = rows * (cols / 256);
		byte[] raw = new byte[blocks * Q4K_BLOCK_BYTES];
		r.nextBytes(raw);
		short scale = Float.floatToFloat16(0.01f);
		for (int b = 0; b < blocks; b++) {
			int off = b * Q4K_BLOCK_BYTES;
			raw[off] = (byte) scale;
			raw[off + 1] = (byte) (scale >> 8);
			raw[off + 2] = (byte) scale;
			raw[off + 3] = (byte) (scale >> 8);
		}
		return raw;
	}
}
