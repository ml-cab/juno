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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.KvRowCodec;
import cab.ml.juno.kvcache.KvRows;
import cab.ml.juno.kvcache.PagedKvArena;
import cab.ml.juno.kvcache.SessionKvTensor;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * A whole-layer prefill window writes the host KV from the device copy's encoded rows: for a side whose device format
 * is that side's host type ({@code q8_0} or {@code q4_0}), the window's stored blocks are downloaded into the host
 * tensor as they are, and the CPU encodes nothing. Checked on a synthetic layer (device RoPE and the attention kernel,
 * so the whole layer runs in the region), two windows (the second starting where the first ended, mid-page on the paged
 * tensor), every pair of device formats, dense and paged host tensors:
 * <ul>
 * <li>the host tensors hold exactly the bytes the FP32 rows give when written as floats (the path before: the rows
 * downloaded as FP32 and encoded by the host codec), the same run done both ways;</li>
 * <li>a side whose device format is its host type is downloaded as stored blocks
 * ({@code materialize(prefill k encoded)}), {@code count x rowBytes} bytes, and no FP32 rows of it are; an {@code f16}
 * side, and a quantized host side the device holds as FP16, download FP32 rows as before.</li>
 * </ul>
 */
@Tag("gpu")
@DisplayName("Prefill-window region - the host KV takes the device copy's encoded rows")
class PrefillRegionEncodedKvTest {

	private static final int H = 512;
	private static final int I = 1024;
	private static final int HEADS = 8;
	private static final int KV_HEADS = 4;
	private static final int HEAD_DIM = 64;
	private static final int KV_DIM = KV_HEADS * HEAD_DIM;
	private static final int W1 = 40;
	private static final int W2 = 24;
	private static final int PAGE = 16;
	private static final PrefillWindowRegion.Shape SHAPE = new PrefillWindowRegion.Shape(H, HEADS * HEAD_DIM, KV_DIM, I,
			HEADS, KV_HEADS, HEAD_DIM, HEADS / KV_HEADS, 1e-5f);

	private static final KvElementType F16 = KvElementType.F16;
	private static final KvElementType F32 = KvElementType.F32;
	private static final KvElementType Q8 = KvElementType.Q8_0;
	private static final KvElementType Q4 = KvElementType.Q4_0;

	private static GpuContext ctx;
	private CudaMatVec mv;
	private PrefillWindowRegion region;
	private final List<AutoCloseable> owned = new ArrayList<>();

	@TempDir
	Path tmp;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
		assumeTrue(GqaAttentionKvKernel.tryLoad() != null, "quantized attention kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@BeforeEach
	void build() {
		mv = new CudaMatVec(ctx);
		PrefillWindowRegion.Layer layer = PrefillWindowRegion.Layer.separate(half(SHAPE.qDim(), H, 1),
				half(KV_DIM, H, 2), half(KV_DIM, H, 3), half(H, SHAPE.qDim(), 4), half(I, H, 5), half(I, H, 6),
				half(H, I, 7), norm(8), norm(9), null, null, null);
		region = PrefillWindowRegion.create("test", mv, SHAPE, new PrefillWindowRegion.Layer[] { layer },
				RopePairing.SPLIT_HALF, 10000f, true);
		assumeTrue(region != null, "prefill-window region unavailable");
		owned.add(region);
	}

	@AfterEach
	void release() throws Exception {
		for (int i = owned.size() - 1; i >= 0; i--)
			owned.get(i).close();
		owned.clear();
		mv.releaseScratch();
	}

	@Test
	@DisplayName("dense host tensors: the float write's bytes, every device format pair")
	void dense() throws Exception {
		for (Case c : cases())
			check(c, t -> new DenseKvTensor(t, KV_DIM));
	}

	@Test
	@DisplayName("paged host tensors: the float write's bytes, every device format pair")
	void paged() throws Exception {
		for (Case c : cases())
			check(c, t -> new PagedKvArena(PAGE, KV_DIM, t, t).newK());
	}

	/** The device formats, and the host types the request's KV has. */
	private record Case(DeviceKvFormats formats, KvElementType hostK, KvElementType hostV) {

		String label() {
			return formats.summary() + " on the device, host K " + hostK.cliName() + ", V " + hostV.cliName();
		}

		boolean encodedK() {
			return formats.k().hostType() == hostK;
		}

		boolean encodedV() {
			return formats.v().hostType() == hostV;
		}
	}

	private static List<Case> cases() {
		return List.of(new Case(formats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q8_0), Q8, Q8),
				new Case(formats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0), Q8, Q4),
				new Case(formats(DeviceKvFormat.Q4_0, DeviceKvFormat.Q4_0), Q4, Q4),
				new Case(formats(DeviceKvFormat.F16, DeviceKvFormat.Q8_0), F16, Q8),
				new Case(formats(DeviceKvFormat.Q4_0, DeviceKvFormat.F16), Q4, F16),
				// A quantized host KV the device holds as FP16 (as for a head width not a multiple of 32).
				new Case(DeviceKvFormats.F16, Q8, Q4),
				// FP16 device rows are an f16 host's rows: taken as they are.
				new Case(DeviceKvFormats.F16, F16, F16),
				// An f32 host holds floats the device does not: FP32 rows come down.
				new Case(DeviceKvFormats.F16, F32, F32),
				new Case(formats(DeviceKvFormat.F16, DeviceKvFormat.Q8_0), F32, Q8));
	}

	private static DeviceKvFormats formats(DeviceKvFormat k, DeviceKvFormat v) {
		return new DeviceKvFormats(k, v);
	}

	/**
	 * Runs the same two windows twice, each time on a new device copy: once with the host tensors written from the
	 * FP32 rows as floats (the reference), once with the host tensors handed to the region, under a recording.
	 */
	private void check(Case c, Function<KvElementType, SessionKvTensor> make) throws Exception {
		float[][] x1 = rows(W1, H, 11);
		float[][] x2 = rows(W2, H, 12);

		SessionKvTensor refK = make.apply(c.hostK()), refV = make.apply(c.hostV());
		try (DeviceKvCache mirror = new DeviceKvCache(ctx, KV_DIM, DeviceKvCache.INITIAL_SEQ_CAPACITY, c.formats())) {
			floatWindow(mirror, x1, 0, refK, refV);
			floatWindow(mirror, x2, W1, refK, refV);
		}

		SessionKvTensor k = make.apply(c.hostK()), v = make.apply(c.hostV());
		List<RecordedEvent> events;
		try (DeviceKvCache mirror = new DeviceKvCache(ctx, KV_DIM, DeviceKvCache.INITIAL_SEQ_CAPACITY, c.formats())) {
			events = record(() -> {
				hostWindow(mirror, x1, 0, k, v);
				hostWindow(mirror, x2, W1, k, v);
			});
		}

		int len = W1 + W2;
		assertThat(stored(k, len)).as(c.label() + ": K").containsExactly(stored(refK, len));
		assertThat(stored(v, len)).as(c.label() + ": V").containsExactly(stored(refV, len));
		checkSide(events, "prefill k", c.encodedK(), c.hostK(), c.label() + ": K");
		checkSide(events, "prefill v", c.encodedV(), c.hostV(), c.label() + ": V");
	}

	private static void checkSide(List<RecordedEvent> events, String name, boolean encoded, KvElementType host,
			String label) {
		long rows = W1 + W2;
		String plain = "materialize(" + name + ")";
		String stored = "materialize(" + name + " encoded)";
		if (encoded) {
			assertThat(sum(events, stored, "copies")).as(label + ": stored-row downloads").isEqualTo(2);
			assertThat(sum(events, stored, "bytes")).as(label + ": stored-row bytes")
					.isEqualTo(rows * KvRowCodec.rowBytes(host, KV_DIM));
			assertThat(sum(events, plain, "copies")).as(label + ": FP32 downloads").isZero();
		} else {
			assertThat(sum(events, stored, "copies")).as(label + ": stored-row downloads").isZero();
			assertThat(sum(events, plain, "bytes")).as(label + ": FP32 bytes").isEqualTo(rows * KV_DIM * Float.BYTES);
		}
	}

	/** One window whose K and V come back as FP32 rows, written into the host tensors as floats. */
	private void floatWindow(DeviceKvCache mirror, float[][] x, int startPos, SessionKvTensor k, SessionKvTensor v) {
		int w = x.length;
		float[][] q = new float[w][SHAPE.qDim()], kOut = new float[w][KV_DIM], vOut = new float[w][KV_DIM];
		try (PrefillWindowRegion.Window win = region.open(w)) {
			assertThat(win.runLayer(0, copy(x), startPos, mirror, q, kOut, vOut)).as("whole layer").isTrue();
		}
		for (int b = 0; b < w; b++) {
			k.writeToken(startPos + b, kOut[b]);
			v.writeToken(startPos + b, vOut[b]);
		}
		mirror.markWritten(startPos, w);
	}

	/** One window that writes the host tensors itself. */
	private void hostWindow(DeviceKvCache mirror, float[][] x, int startPos, SessionKvTensor k, SessionKvTensor v) {
		int w = x.length;
		float[][] q = new float[w][SHAPE.qDim()], kOut = new float[w][KV_DIM], vOut = new float[w][KV_DIM];
		try (PrefillWindowRegion.Window win = region.open(w)) {
			assertThat(win.runLayer(0, copy(x), startPos, mirror, q, kOut, vOut, k, v)).as("whole layer").isTrue();
		}
		mirror.markWritten(startPos, w);
	}

	/** The stored bytes of positions {@code [0, len)}: the encoded rows, or the floats' bits for {@code f32}. */
	private static byte[] stored(SessionKvTensor t, int len) {
		KvRows view = t.attentionRows(len);
		int bpr = view.bytesPerRow();
		ByteBuffer out = ByteBuffer.allocate(len * bpr).order(ByteOrder.LITTLE_ENDIAN);
		for (int p = 0; p < len; p++) {
			int page = p / view.rowsPerPage();
			int slot = p % view.rowsPerPage();
			if (t.type().storesFloats())
				for (int i = 0; i < KV_DIM; i++)
					out.putInt(Float.floatToRawIntBits(view.floatPage(page)[slot * KV_DIM + i]));
			else
				out.put(view.bytePage(page), slot * bpr, bpr);
		}
		return out.array();
	}

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("encoded-kv-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceStaging").withThreshold(Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		return RecordingFile.readAllEvents(jfr);
	}

	private static long sum(List<RecordedEvent> events, String site, String field) {
		return events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
				.filter(e -> "D2H".equals(e.getString("direction")) && site.equals(e.getString("site")))
				.mapToLong(e -> e.getLong(field)).sum();
	}

	private PrefillWindowRegion.Matrix half(int rows, int cols, long seed) {
		Random r = new Random(seed);
		float[] w = new float[rows * cols];
		for (int i = 0; i < w.length; i++)
			w[i] = (r.nextFloat() * 2f - 1f) * 0.05f;
		DeviceHalfMatrix m = mv.uploadHalf(w, rows, cols);
		owned.add(m);
		return new PrefillWindowRegion.Matrix(null, m);
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
}
