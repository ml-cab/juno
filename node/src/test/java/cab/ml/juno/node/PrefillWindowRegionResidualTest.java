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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The prefill-window region keeps the residual stream on the device from layer to
 * layer: it is uploaded once per window and downloaded once, when the caller asks
 * for it ({@link PrefillWindowRegion.Window#materializeResidual}), instead of once
 * each way per layer. Checked on a synthetic model through the split form of a
 * layer (the caller stands in for attention):
 * <ul>
 * <li>three layers with the residual kept on the device give the same bits as the
 * same layers with the residual downloaded and uploaded again after each one, with
 * one residual upload and one download for the window instead of three of each;
 * <li>a device out-of-memory error inside a layer whose input exists only on the
 * device - after the layer has started updating the residual in place, and before
 * - still hands the caller that input exactly
 * ({@link PrefillWindowRegion.Window#recoverLayerInput}), so the layer can be
 * redone on the host path, and the window carries on correctly afterwards.
 * </ul>
 */
@Tag("gpu")
@DisplayName("Prefill-window region - the residual stays on the device across layers")
class PrefillWindowRegionResidualTest {

	private static final int H = 1024;
	private static final int I = 8192;
	private static final int HEADS = 16;
	private static final int HEAD_DIM = 64;
	private static final int W = 40;
	private static final int Q4K_BLOCK_BYTES = 144;
	private static final PrefillWindowRegion.Shape SHAPE = new PrefillWindowRegion.Shape(H, H, H, I, HEADS, HEADS,
			HEAD_DIM, 1, 1e-5f);

	private static GpuContext ctx;
	private CudaMatVec mv;
	private final List<AutoCloseable> owned = new ArrayList<>();

	@TempDir
	Path tmp;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@BeforeEach
	void backend() {
		mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "K-quant MMQ kernel unavailable");
	}

	@AfterEach
	void release() throws Exception {
		for (int i = owned.size() - 1; i >= 0; i--)
			owned.get(i).close();
		owned.clear();
		mv.releaseScratch();
	}

	@Test
	@DisplayName("kept on the device for three layers: same bits as a round trip per layer, one copy each way")
	void residentAcrossLayers_matchesRoundTripPerLayer() throws Exception {
		PrefillWindowRegion region = region(halfLayer(10), q4kLayer(20), halfLayer(30));
		float[][] input = rows(W, H, 1);

		float[][] resident = copy(input);
		List<RecordedEvent> residentEvents = record(() -> {
			try (PrefillWindowRegion.Window win = region.open(W)) {
				for (int li = 0; li < 3; li++)
					runSplit(win, li, resident);
				win.materializeResidual(resident);
			}
		});

		float[][] roundTrip = copy(input);
		List<RecordedEvent> roundTripEvents = record(() -> {
			try (PrefillWindowRegion.Window win = region.open(W)) {
				for (int li = 0; li < 3; li++) {
					runSplit(win, li, roundTrip);
					win.materializeResidual(roundTrip);
				}
			}
		});

		for (int b = 0; b < W; b++)
			assertThat(resident[b]).as("row " + b).containsExactly(roundTrip[b]);
		assertThat(copies(residentEvents, "H2D", "upload(prefill residual)")).as("residual uploads, kept resident")
				.isEqualTo(1);
		assertThat(copies(residentEvents, "D2H", "materialize(prefill residual)"))
				.as("residual downloads, kept resident").isEqualTo(1);
		assertThat(copies(roundTripEvents, "H2D", "upload(prefill residual)")).as("residual uploads, round trip")
				.isEqualTo(3);
		assertThat(copies(roundTripEvents, "D2H", "materialize(prefill residual)"))
				.as("residual downloads, round trip").isEqualTo(3);
	}

	@Test
	@DisplayName("out of device memory after the layer has updated the residual: the layer's input is recovered")
	void oomAfterResidualUpdate_recoversLayerInput() {
		// Layer 1's gate, up and down are packed K-quant: their first GEMM grows the backend's
		// Q8_1 window scratch, after the output projection has already been added to the residual.
		PrefillWindowRegion region = region(halfLayer(10), splitFfnQ4kLayer(40));
		assertRecoversLayerOneInput(region, false);
	}

	@Test
	@DisplayName("out of device memory before the layer has touched the residual: the layer's input is recovered")
	void oomBeforeResidualUpdate_recoversLayerInput() {
		// Layer 1's Q/K/V is one packed K-quant matrix: its GEMM is the layer's first and fails.
		PrefillWindowRegion region = region(halfLayer(10), fusedQkvQ4kLayer(50));
		assertRecoversLayerOneInput(region, true);
	}

	/**
	 * Layer 0 on the device, then layer 1 with no device memory left, so its first packed
	 * K-quant GEMM fails growing the backend's Q8_1 window scratch (layer 0 has FP16
	 * weights and never grew it); the recovered rows must equal layer 0's output, and after
	 * the memory is back the window must run layer 1 to the same result as a clean run.
	 */
	private void assertRecoversLayerOneInput(PrefillWindowRegion region, boolean failsInRunLayer) {
		float[][] input = rows(W, H, 2);

		float[][] layerZero = copy(input);
		try (PrefillWindowRegion.Window win = region.open(W)) {
			runSplit(win, 0, layerZero);
			win.materializeResidual(layerZero);
		}

		float[][] x = copy(input);
		float[][] q = new float[W][H], k = new float[W][H], v = new float[W][H];
		try (PrefillWindowRegion.Window win = region.open(W)) {
			runSplit(win, 0, x);
			// No headroom: the scratch the first packed GEMM grows is a window's Q8_1 copy, 46 KB
			// here, not the whole FP16 weight matrix the dequantizing route needed.
			List<MemorySegment> ballast = fillDevice(0);
			try {
				assertThatThrownBy(() -> {
					if (failsInRunLayer) {
						win.runLayer(1, x, 0, null, q, k, v);
					} else {
						assertThat(win.runLayer(1, x, 0, null, q, k, v)).isFalse();
						win.finishLayer(1, q, x);
					}
				}).isInstanceOf(IllegalStateException.class)
						.matches(ex -> GpuLayerOffload.isVramOom((IllegalStateException) ex), "is a VRAM OOM");
				win.recoverLayerInput(x);
			} finally {
				for (MemorySegment s : ballast)
					ctx.bindings().deviceFree(s);
			}
			for (int b = 0; b < W; b++)
				assertThat(x[b]).as("recovered input, row " + b).containsExactly(layerZero[b]);

			runSplit(win, 1, x);
			win.materializeResidual(x);
		}

		float[][] clean = copy(input);
		try (PrefillWindowRegion.Window win = region.open(W)) {
			runSplit(win, 0, clean);
			runSplit(win, 1, clean);
			win.materializeResidual(clean);
		}
		for (int b = 0; b < W; b++)
			assertThat(x[b]).as("layer 1 after recovery, row " + b).containsExactly(clean[b]);
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	/** One layer in the split form, with the caller's attention standing in as "the output is Q". */
	private static void runSplit(PrefillWindowRegion.Window win, int li, float[][] x) {
		float[][] q = new float[x.length][H], k = new float[x.length][H], v = new float[x.length][H];
		assertThat(win.runLayer(li, x, 0, null, q, k, v)).as("split form (no RoPE on the device)").isFalse();
		win.finishLayer(li, q, x);
	}

	private PrefillWindowRegion region(PrefillWindowRegion.Layer... layers) {
		PrefillWindowRegion r = PrefillWindowRegion.create("test", mv, SHAPE, layers, null, 0f, false);
		assertThat(r).as("region").isNotNull();
		owned.add(r);
		return r;
	}

	private PrefillWindowRegion.Layer halfLayer(long seed) {
		return PrefillWindowRegion.Layer.separate(half(H, H, seed), half(H, H, seed + 1), half(H, H, seed + 2),
				half(H, H, seed + 3), half(I, H, seed + 4), half(I, H, seed + 5), half(H, I, seed + 6), norm(seed + 7),
				norm(seed + 8), null, null, null);
	}

	private PrefillWindowRegion.Layer q4kLayer(long seed) {
		return PrefillWindowRegion.Layer.separate(q4k(H, H, seed), q4k(H, H, seed + 1), q4k(H, H, seed + 2),
				q4k(H, H, seed + 3), q4k(I, H, seed + 4), q4k(I, H, seed + 5), q4k(H, I, seed + 6), norm(seed + 7),
				norm(seed + 8), null, null, null);
	}

	private PrefillWindowRegion.Layer splitFfnQ4kLayer(long seed) {
		return PrefillWindowRegion.Layer.separate(half(H, H, seed), half(H, H, seed + 1), half(H, H, seed + 2),
				half(H, H, seed + 3), q4k(I, H, seed + 4), q4k(I, H, seed + 5), q4k(H, I, seed + 6), norm(seed + 7),
				norm(seed + 8), null, null, null);
	}

	private PrefillWindowRegion.Layer fusedQkvQ4kLayer(long seed) {
		return PrefillWindowRegion.Layer.mixed(q4k(3 * H, H, seed), null, null, null, half(H, H, seed + 3), null,
				half(I, H, seed + 4), half(I, H, seed + 5), half(H, I, seed + 6), norm(seed + 7), norm(seed + 8));
	}

	private PrefillWindowRegion.Matrix half(int rows, int cols, long seed) {
		DeviceHalfMatrix m = mv.uploadHalf(weights(rows * cols, seed), rows, cols);
		owned.add(m);
		return new PrefillWindowRegion.Matrix(null, m);
	}

	private PrefillWindowRegion.Matrix q4k(int rows, int cols, long seed) {
		DeviceQ4KMatrix m = mv.uploadQ4K(randomQ4K(rows, cols, seed), rows, cols);
		owned.add(m);
		return new PrefillWindowRegion.Matrix(m, null);
	}

	/** Allocates device memory in shrinking chunks until about {@code leave} bytes are free. */
	private static List<MemorySegment> fillDevice(long leave) {
		GpuBindings gpu = ctx.bindings();
		List<MemorySegment> held = new ArrayList<>();
		long chunk = 256L << 20;
		while (chunk >= (4L << 10)) {
			long free = gpu.memGetInfo(ctx.deviceIndex())[0];
			if (free - chunk < leave) {
				chunk >>= 1;
				continue;
			}
			try {
				held.add(gpu.deviceMalloc(ctx.deviceIndex(), chunk));
			} catch (IllegalStateException ex) {
				chunk >>= 1;
			}
		}
		return held;
	}

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("residual-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceStaging").withThreshold(Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		return RecordingFile.readAllEvents(jfr);
	}

	private static long copies(List<RecordedEvent> events, String direction, String site) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> direction.equals(e.getString("direction")) && site.equals(e.getString("site")))
				.mapToLong(e -> e.getLong("copies")).sum();
	}

	private static float[] weights(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (r.nextFloat() * 2f - 1f) * 0.05f;
		return v;
	}

	private static float[] norm(long seed) {
		Random r = new Random(seed);
		float[] v = new float[H];
		for (int i = 0; i < H; i++)
			v[i] = 0.5f + r.nextFloat();
		return v;
	}

	private static float[][] rows(int n, int dim, long seed) {
		Random r = new Random(seed);
		float[][] x = new float[n][dim];
		for (float[] row : x)
			for (int i = 0; i < dim; i++)
				row[i] = r.nextFloat() * 2f - 1f;
		return x;
	}

	private static float[][] copy(float[][] x) {
		float[][] c = new float[x.length][];
		for (int b = 0; b < x.length; b++)
			c[b] = x[b].clone();
		return c;
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
