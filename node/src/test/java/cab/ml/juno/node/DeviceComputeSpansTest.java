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
 * {@code juno.DeviceCompute} on a real CUDA device: each prefill-width kernel (the tiled
 * GEMMs, the strided batched GEMV, the FP32 BLAS GEMM, the attention kernel) is totalled
 * under its site with a measured duration, a decode-width packed GEMV is counted untimed,
 * the host FP16 packing of an activation window is a {@code HOST} staging site apart from
 * the bus copies, and recording changes no output.
 */
@Tag("gpu")
@DisplayName("juno.DeviceCompute on CUDA")
class DeviceComputeSpansTest {

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
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a 32-row FP16 GEMM: one gemm_half kernel timed on the device, and the host packing apart")
	void fp16Gemm_isTimedAndPackingIsHostWork() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 1), ROWS, COLS);
		float[][] x = randomRows(32, COLS, 2);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		RecordedEvent gemm = compute(events, "gemm_half", "prefill");
		assertThat(gemm.getLong("count")).isEqualTo(1);
		assertThat(gemm.getLong("timedCount")).isEqualTo(1);
		assertThat(gemm.getLong("computeNanos")).isPositive();

		RecordedEvent pack = staging(events, "pack_fp16_host", "prefill");
		assertThat(pack.getString("direction")).isEqualTo("HOST");
		assertThat(pack.getLong("bytes")).isEqualTo((long) COLS * 32 * Short.BYTES);
		assertThat(pack.getLong("timedCopies")).isEqualTo(1);
		assertThat(pack.getLong("transferNanos")).isPositive();
	}

	@Test
	@DisplayName("a batched Q4_K GEMM: the GEMM is gemm_half, apart from its dequantization")
	void q4kGemm_isGemmHalf() throws Exception {
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 3), ROWS, COLS);
		float[][] x = randomRows(16, COLS, 4);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		RecordedEvent gemm = compute(events, "gemm_half", "prefill");
		assertThat(gemm.getLong("timedCount")).isEqualTo(1);
		assertThat(gemm.getLong("computeNanos")).isPositive();
	}

	@Test
	@DisplayName("a 4-row FP16 batch: the strided batched GEMV is timed")
	void smallFp16Batch_isGemvHalfBatched() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 5), ROWS, COLS);
		float[][] x = randomRows(4, COLS, 6);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		RecordedEvent gemv = compute(events, "gemv_half_batched", "prefill");
		assertThat(gemv.getLong("timedCount")).isEqualTo(1);
		assertThat(gemv.getLong("computeNanos")).isPositive();
	}

	@Test
	@DisplayName("a 16-row FP32 GEMM: gemm_fp32 is timed")
	void fp32Gemm_isTimed() throws Exception {
		DeviceFloatMatrix a = mv.upload(randomFloats(ROWS * COLS, 7), ROWS, COLS);
		float[][] x = randomRows(16, COLS, 8);
		List<RecordedEvent> events = record(() -> mv.sgemm(a, x));
		a.close();

		RecordedEvent gemm = compute(events, "gemm_fp32", "prefill");
		assertThat(gemm.getLong("timedCount")).isEqualTo(1);
		assertThat(gemm.getLong("computeNanos")).isPositive();
	}

	@Test
	@DisplayName("juno.MatVec carries its batch width: a GEMM its window, a GEMV one row")
	void matVec_recordsTheCallsBatchWidth() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 11), ROWS, COLS);
		float[][] x = randomRows(32, COLS, 12);
		float[][] small = randomRows(4, COLS, 13);
		List<RecordedEvent> events = record(() -> {
			mv.sgemm(a, x);
			mv.sgemm(a, small);
			mv.sgemv(a, x[0]);
		});
		a.close();

		List<Integer> widths = events.stream().filter(e -> e.getEventType().getName().equals("juno.MatVec"))
				.map(e -> e.hasField("windowSize") ? e.getInt("windowSize") : -1).toList();
		assertThat(widths).containsExactly(32, 4, 1);
	}

	@Test
	@DisplayName("decode-width packed GEMVs are counted, untimed")
	void decodeQ4kGemv_isCountedNotTimed() throws Exception {
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		DeviceQ4KMatrix a = mv.uploadQ4K(randomQ4K(ROWS, COLS, 9), ROWS, COLS);
		float[] x = randomFloats(COLS, 10);
		List<RecordedEvent> events = record(() -> {
			mv.sgemv(a, x);
			mv.sgemv(a, x);
		});
		a.close();

		RecordedEvent mmq = compute(events, "mmq_packed", "decode");
		assertThat(mmq.getLong("count")).isEqualTo(2);
		assertThat(mmq.getLong("timedCount")).isZero();
	}

	@Test
	@DisplayName("a prefill-width attention launch: gqa_attention is timed")
	void attention_isTimed() throws Exception {
		CudaGqaAttention attn = CudaGqaAttention.tryCreate(ctx);
		assumeTrue(attn != null && GqaAttentionKernel.tryLoad() != null, "GQA kernel unavailable");
		int heads = 4, kvHeads = 2, headDim = 64, kvDim = kvHeads * headDim, window = 8;
		DeviceKvCache[] layers = attn.newLayers(1, kvDim);
		try {
			DeviceKvCache kv = layers[0];
			for (int p = 0; p < window; p++)
				kv.appendToken(p, randomFloats(kvDim, 100 + p), randomFloats(kvDim, 200 + p), window);
			DeviceKvCache[] kvs = new DeviceKvCache[window];
			int[] seqLens = new int[window];
			float[][] q = randomRows(window, heads * headDim, 11);
			for (int b = 0; b < window; b++) {
				kvs[b] = kv;
				seqLens[b] = b + 1;
			}
			float[][] out = new float[window][];
			List<RecordedEvent> events = record(
					() -> attn.attendBatched(kvs, q, seqLens, out, heads, headDim, heads / kvHeads, kvDim));

			RecordedEvent gqa = compute(events, "gqa_attention", "prefill");
			assertThat(gqa.getLong("timedCount")).isEqualTo(1);
			assertThat(gqa.getLong("computeNanos")).isPositive();
		} finally {
			for (DeviceKvCache l : layers)
				l.close();
		}
	}

	@Test
	@DisplayName("recording changes no output: a 32-row GEMM is bit-identical with and without it")
	void recording_isBitIdentical() throws Exception {
		DeviceHalfMatrix a = mv.uploadHalf(randomFloats(ROWS * COLS, 12), ROWS, COLS);
		float[][] x = randomRows(32, COLS, 13);
		float[][] plain = mv.sgemm(a, x);
		float[][][] recorded = new float[1][][];
		record(() -> recorded[0] = mv.sgemm(a, x));
		a.close();
		for (int b = 0; b < x.length; b++)
			assertThat(recorded[0][b]).as("row " + b).containsExactly(plain[b]);
	}

	// ── helpers ──────────────────────────────────────────────────────────────

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("compute-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceCompute").withThreshold(java.time.Duration.ZERO);
			rec.enable("juno.DeviceStaging").withThreshold(java.time.Duration.ZERO);
			rec.enable("juno.MatVec").withThreshold(java.time.Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		assertThat(Files.size(jfr)).isPositive();
		return RecordingFile.readAllEvents(jfr);
	}

	private static RecordedEvent compute(List<RecordedEvent> events, String site, String phase) {
		return only(events, "juno.DeviceCompute", site, phase);
	}

	private static RecordedEvent staging(List<RecordedEvent> events, String site, String phase) {
		return only(events, "juno.DeviceStaging", site, phase);
	}

	private static RecordedEvent only(List<RecordedEvent> events, String name, String site, String phase) {
		List<RecordedEvent> matching = events.stream().filter(e -> e.getEventType().getName().equals(name))
				.filter(e -> site.equals(e.getString("site")) && phase.equals(e.getString("phase"))).toList();
		assertThat(matching).as(name + " " + site + " / " + phase).hasSize(1);
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
