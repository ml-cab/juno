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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvRowCodec;

/**
 * A device KV stored as {@code q8_0} or {@code q4_0} blocks: what the device writes is the host codec's bytes, and
 * what GPU attention reads over it matches the CPU attention over the same host codec within the bound the FP16
 * kernel is held to ({@link GqaAttentionKernelParityTest}).
 */
@Tag("gpu")
@DisplayName("Quantized device KV - device writes equal the host codecs, attention matches the CPU over them")
class DeviceKvQuantParityTest {

	private static final float TOL = 3e-3f;
	private static final float TOL_LONG = 8e-3f;

	private static final DeviceKvFormats Q8 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q8_0);
	private static final DeviceKvFormats Q8_Q4 = new DeviceKvFormats(DeviceKvFormat.Q8_0, DeviceKvFormat.Q4_0);
	private static final DeviceKvFormats Q4 = new DeviceKvFormats(DeviceKvFormat.Q4_0, DeviceKvFormat.Q4_0);
	private static final DeviceKvFormats F16_Q8 = new DeviceKvFormats(DeviceKvFormat.F16, DeviceKvFormat.Q8_0);
	private static final DeviceKvFormats Q4_F16 = new DeviceKvFormats(DeviceKvFormat.Q4_0, DeviceKvFormat.F16);
	private static final DeviceKvFormats[] QUANTIZED = { Q8, Q8_Q4, Q4, F16_Q8, Q4_F16 };

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(GqaAttentionKernel.isAvailable(), "Skipping - GPU attention kernel unavailable");
		ctx = GpuContext.init(0);
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
		assumeTrue(GqaAttentionKvKernel.tryLoad() != null, "quantized attention kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	private record Shape(String name, int numHeads, int numKvHeads, int headDim) {
		int kvDim() {
			return numKvHeads * headDim;
		}

		int gqaRatio() {
			return numHeads / numKvHeads;
		}
	}

	private static final Shape[] SHAPES = { new Shape("TinyLlama", 32, 4, 64), new Shape("Phi-3.5-mini", 32, 32, 96),
			new Shape("Mistral 7B", 32, 8, 128), new Shape("head 256", 8, 2, 256) };

	// ---- device writes --------------------------------------------------------------------------------------------

	@Test
	@DisplayName("a device window write stores the host codec's bytes, rounding ties and degenerate blocks included")
	void deviceWindowWrite_isTheHostEncoding() {
		int kvDim = 256;
		int rows = 70;
		float[][] k = edgeRows(rows, kvDim, 1);
		float[][] v = edgeRows(rows, kvDim, 2);
		PrefillWindowKernels kernels = PrefillWindowKernels.tryLoad();
		for (DeviceKvFormats f : new DeviceKvFormats[] { Q8, Q8_Q4, Q4, F16_Q8, Q4_F16, DeviceKvFormats.F16 }) {
			try (DeviceKvCache cache = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f);
					ResidentChain chain = ResidentChain.open(ctx)) {
				ResidentActivation dk = chain.allocate(rows, kvDim);
				ResidentActivation dv = chain.allocate(rows, kvDim);
				dk.upload(k);
				dv.upload(v);
				chain.sync();
				// Two windows, the second past the initial capacity, so the write lands after a grow.
				cache.writeWindowOnDevice(0, 30, dk.devicePointer(), dv.devicePointer(), kernels, chain.stream());
				chain.sync();
				cache.markWritten(0, 30);
				long rowBytes = 4L * kvDim;
				cache.writeWindowOnDevice(30, rows - 30, dk.devicePointer().asSlice(30 * rowBytes),
						dv.devicePointer().asSlice(30 * rowBytes), kernels, chain.stream());
				chain.sync();
				cache.markWritten(30, rows - 30);
				assertThat(cache.rawK(rows)).as(f.summary() + " K").containsExactly(hostBytes(f.k(), k, kvDim));
				assertThat(cache.rawV(rows)).as(f.summary() + " V").containsExactly(hostBytes(f.v(), v, kvDim));
			}
		}
	}

	@Test
	@DisplayName("the decode batch's append stores each request's row as the window write does")
	void batchAppend_isTheHostEncoding() {
		int kvDim = 1024;
		int[] positions = { 0, 63, 64, 300 };
		int n = positions.length;
		float[][] k = edgeRows(n, kvDim, 3);
		float[][] v = edgeRows(n, kvDim, 4);
		BatchedDecodeKernels batched = BatchedDecodeKernels.tryLoad();
		assumeTrue(batched != null, "decode batch kernels failed to load");
		GpuBindings gpu = ctx.bindings();
		for (DeviceKvFormats f : QUANTIZED) {
			DeviceKvCache[] caches = new DeviceKvCache[n];
			MemorySegment dTables = MemorySegment.NULL;
			try (ResidentChain chain = ResidentChain.open(ctx); Arena arena = Arena.ofConfined()) {
				for (int b = 0; b < n; b++) {
					caches[b] = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f);
					caches[b].ensureCapacity(positions[b]);
				}
				ResidentActivation dk = chain.allocate(n, kvDim);
				ResidentActivation dv = chain.allocate(n, kvDim);
				dk.upload(k);
				dv.upload(v);
				long ptrBytes = n * ADDRESS.byteSize();
				long tableBytes = 2 * ptrBytes + (long) n * Integer.BYTES;
				MemorySegment host = arena.allocate(tableBytes, 8);
				for (int b = 0; b < n; b++) {
					host.setAtIndex(ADDRESS, b, caches[b].kPointer());
					host.setAtIndex(ADDRESS, n + b, caches[b].vPointer());
					host.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, positions[b] + 1);
				}
				dTables = gpu.deviceMalloc(ctx.deviceIndex(), tableBytes);
				DeviceStaging.copy(gpu, dTables, host, tableBytes, GpuBindings.H2D, 0, "test tables");
				chain.sync();
				batched.kvAppendRows(dk.devicePointer(), dv.devicePointer(), dTables, dTables.asSlice(ptrBytes),
						dTables.asSlice(2 * ptrBytes), n, kvDim, f, chain.stream());
				chain.sync();
				for (int b = 0; b < n; b++) {
					int p = positions[b];
					byte[] gotK = rowOf(caches[b].rawK(p + 1), f.k().rowBytes(kvDim), p);
					byte[] gotV = rowOf(caches[b].rawV(p + 1), f.v().rowBytes(kvDim), p);
					assertThat(gotK).as(f.summary() + " K of request " + b)
							.containsExactly(hostBytes(f.k(), new float[][] { k[b] }, kvDim));
					assertThat(gotV).as(f.summary() + " V of request " + b)
							.containsExactly(hostBytes(f.v(), new float[][] { v[b] }, kvDim));
				}
			} finally {
				if (dTables.address() != 0)
					gpu.deviceFree(dTables);
				for (DeviceKvCache c : caches)
					if (c != null)
						c.close();
			}
		}
	}

	// ---- the cache ------------------------------------------------------------------------------------------------

	@Test
	@DisplayName("host appends pack the host codec's bytes; a grow keeps them; download reads the host codec's values")
	void hostAppends_roundTrip() {
		int kvDim = 512;
		int rows = 200;
		float[][] k = edgeRows(rows, kvDim, 5);
		float[][] v = edgeRows(rows, kvDim, 6);
		for (DeviceKvFormats f : QUANTIZED) {
			try (DeviceKvCache perToken = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f);
					DeviceKvCache window = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f);
					DeviceKvCache prefix = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f)) {
				for (int p = 0; p < rows; p++)
					perToken.appendToken(p, k[p], v[p]);
				window.appendWindow(0, slice(k, 0, 50), slice(v, 0, 50), 50);
				window.appendWindow(50, slice(k, 50, rows), slice(v, 50, rows), rows - 50);
				prefix.replacePrefix(flatten(k), flatten(v), rows);
				byte[] wantK = hostBytes(f.k(), k, kvDim);
				byte[] wantV = hostBytes(f.v(), v, kvDim);
				for (DeviceKvCache c : new DeviceKvCache[] { perToken, window, prefix }) {
					assertThat(c.validTokens()).isEqualTo(rows);
					assertThat(c.rawK(rows)).as(f.summary()).containsExactly(wantK);
					assertThat(c.rawV(rows)).as(f.summary()).containsExactly(wantV);
				}
				assertThat(perToken.downloadK(rows)).containsExactly(hostValues(f.k(), k, kvDim));
				assertThat(perToken.downloadV(rows)).containsExactly(hostValues(f.v(), v, kvDim));
			}
		}
	}

	@Test
	@DisplayName("the cache allocates the quantized size, and gives it back")
	void allocatesTheQuantizedSize() {
		int kvDim = 1024;
		long before = DeviceKvCache.allocatedBytes();
		try (DeviceKvCache c = new DeviceKvCache(ctx, kvDim, 128, Q8_Q4)) {
			assertThat(DeviceKvCache.allocatedBytes() - before).isEqualTo(128L * (32 * 34 + 32 * 18));
			c.ensureCapacity(200);
			assertThat(DeviceKvCache.allocatedBytes() - before).isEqualTo(256L * (32 * 34 + 32 * 18));
		}
		assertThat(DeviceKvCache.allocatedBytes()).isEqualTo(before);
	}

	@Test
	@DisplayName("a quantized cache is not shifted in place (keys would be re-encoded); the caller rewrites it")
	void shiftInPlace_declinesForQuantized() {
		int kvDim = 256;
		float[][] k = edgeRows(40, kvDim, 7);
		float[][] v = edgeRows(40, kvDim, 8);
		RopeShift rope = RopeShift.standard(64, 10000f, RopePairing.ADJACENT);
		try (DeviceKvCache c = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, Q8)) {
			c.appendWindow(0, k, v, 40);
			byte[] before = c.rawK(40);
			assertThat(c.shiftInPlace(4, 10, 40, rope, 4)).isFalse();
			assertThat(c.validTokens()).isEqualTo(40);
			assertThat(c.rawK(40)).containsExactly(before);
		}
	}

	// ---- attention ------------------------------------------------------------------------------------------------

	@Test
	@DisplayName("decode rows over separate caches match the CPU attention over the host codec")
	void decodeStreams_matchTheCpu() {
		int[] positions = { 10, 257, 1500 };
		for (Shape shape : SHAPES)
			for (DeviceKvFormats f : QUANTIZED)
				runStreams(shape, f, positions, 0);
	}

	@Test
	@DisplayName("a prefill window over one cache (tiled, 32 rows a block) matches the CPU attention over the host codec")
	void prefillWindow_matchesTheCpu() {
		for (Shape shape : SHAPES)
			for (DeviceKvFormats f : QUANTIZED)
				runWindow(shape, f, 37, 40, 0);
	}

	@Test
	@DisplayName("a sliding window reads the same keys over a quantized cache, decode and prefill")
	void slidingWindow_matchesTheCpu() {
		Shape shape = SHAPES[2];
		for (DeviceKvFormats f : new DeviceKvFormats[] { Q8, Q8_Q4 }) {
			runWindow(shape, f, 200, 33, 64);
			runStreams(shape, f, new int[] { 30, 700 }, 128);
		}
	}

	@Test
	@DisplayName("a long decode context (8,192 keys) stays within the long-context bound")
	void longContext_matchesTheCpu() {
		runStreams(SHAPES[0], Q8_Q4, new int[] { 8191 }, 0);
	}

	@Test
	@DisplayName("the quantized kernel over FP16 rows: the prefill tile is the FP16 kernel bit for bit, decode within 1e-5")
	void quantizedKernelOverFp16_isTheFp16Kernel() {
		Shape shape = SHAPES[2];
		int kvDim = shape.kvDim();
		int seqLen = 300;
		Random rng = new Random(21);
		float[][] k = gaussianRows(rng, seqLen, kvDim);
		float[][] v = gaussianRows(rng, seqLen, kvDim);
		int batch = 3;
		float[][] q = gaussianRows(rng, batch, shape.numHeads() * shape.headDim());
		int[] seqLens = { 100, 299, 300 };
		GpuBindings gpu = ctx.bindings();
		try (DeviceKvCache c = new DeviceKvCache(ctx, kvDim); Arena arena = Arena.ofConfined()) {
			c.appendWindow(0, k, v, seqLen);
			int rowDim = shape.numHeads() * shape.headDim();
			long qBytes = (long) batch * rowDim * Float.BYTES;
			long ptrBytes = batch * ADDRESS.byteSize();
			MemorySegment dQ = gpu.deviceMalloc(ctx.deviceIndex(), qBytes);
			MemorySegment dOutA = gpu.deviceMalloc(ctx.deviceIndex(), qBytes);
			MemorySegment dOutB = gpu.deviceMalloc(ctx.deviceIndex(), qBytes);
			MemorySegment dTables = gpu.deviceMalloc(ctx.deviceIndex(), 2 * ptrBytes + batch * Integer.BYTES);
			try {
				MemorySegment hq = arena.allocate(qBytes);
				for (int b = 0; b < batch; b++)
					MemorySegment.copy(q[b], 0, hq, JAVA_FLOAT, (long) b * rowDim * Float.BYTES, rowDim);
				DeviceStaging.copy(gpu, dQ, hq, qBytes, GpuBindings.H2D, 0, "test q");
				MemorySegment ht = arena.allocate(2 * ptrBytes + batch * Integer.BYTES, 8);
				for (int b = 0; b < batch; b++) {
					ht.setAtIndex(ADDRESS, b, c.kPointer());
					ht.setAtIndex(ADDRESS, batch + b, c.vPointer());
					ht.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, seqLens[b]);
				}
				DeviceStaging.copy(gpu, dTables, ht, ht.byteSize(), GpuBindings.H2D, 0, "test tables");
				for (int rowsPerBlock : new int[] { 1, 4 }) {
					GqaAttentionKernel.tryLoad().launch(dQ, dTables, dTables.asSlice(ptrBytes),
							dTables.asSlice(2 * ptrBytes), dOutA, batch, shape.numHeads(), shape.gqaRatio(),
							shape.headDim(), kvDim, rowsPerBlock, 0, null);
					GqaAttentionKvKernel.tryLoad().launch(dQ, dTables, dTables.asSlice(ptrBytes),
							dTables.asSlice(2 * ptrBytes), dOutB, batch, shape.numHeads(), shape.gqaRatio(),
							shape.headDim(), kvDim, rowsPerBlock, 0, DeviceKvFormats.F16, null);
					float[] a = download(gpu, dOutA, batch * rowDim, arena);
					float[] b = download(gpu, dOutB, batch * rowDim, arena);
					// The tiled path stages the same rows into the same tiles; a decode row's lanes hold the head
					// block-major over a quantized cache, which reorders each lane's partial sums.
					for (int i = 0; i < a.length; i++) {
						if (rowsPerBlock > 1)
							assertThat(Float.floatToRawIntBits(b[i])).as("rowsPerBlock " + rowsPerBlock + " [" + i + "]")
									.isEqualTo(Float.floatToRawIntBits(a[i]));
						else
							assertThat(b[i]).as("rowsPerBlock 1 [" + i + "]").isCloseTo(a[i], within(1e-5f));
					}
				}
			} finally {
				gpu.deviceFree(dQ);
				gpu.deviceFree(dOutA);
				gpu.deviceFree(dOutB);
				gpu.deviceFree(dTables);
			}
		}
	}

	@Test
	@DisplayName("a prefill tile over quantized rows whose values FP16 holds exactly is the FP16 kernel bit for bit")
	void prefillTileOverExactValues_isTheFp16Kernel() {
		// Every value is q * 2^-k with an integer q the encoder reproduces, so the q8_0 and q4_0 blocks decode to
		// exactly the floats an FP16 cache holds: the tiles a prefill window stages are the same either way, and the
		// arithmetic after staging is shared, so any staging defect (wrong element, block, scale or nibble) shows here
		// as a bit difference.
		int[][] windows = { { 37, 40, 0 }, { 200, 33, 64 }, { 3, 5, 0 } }; // startPos, rows, sliding window
		for (Shape shape : SHAPES) {
			int kvDim = shape.kvDim();
			for (DeviceKvFormats f : QUANTIZED) {
				for (int[] win : windows) {
					int startPos = win[0];
					int w = win[1];
					int rows = startPos + w;
					Random rng = new Random(shape.name().hashCode() * 13 + f.hashCode() + startPos);
					float[][] k = exactRows(rng, rows, kvDim, f.k());
					float[][] v = exactRows(rng, rows, kvDim, f.v());
					assertThat(hostValues(f.k(), k, kvDim)).as("K values survive the host codec").isEqualTo(flatten(k));
					assertThat(hostValues(f.v(), v, kvDim)).as("V values survive the host codec").isEqualTo(flatten(v));
					float[][] q = gaussianRows(rng, w, shape.numHeads() * shape.headDim());
					float[][] fp16Out = attendWindow(shape, DeviceKvFormats.F16, k, v, q, startPos, win[2]);
					float[][] quantOut = attendWindow(shape, f, k, v, q, startPos, win[2]);
					for (int b = 0; b < w; b++)
						for (int i = 0; i < fp16Out[b].length; i++)
							assertThat(Float.floatToRawIntBits(quantOut[b][i]))
									.as("%s %s start %d rows %d window %d row %d [%d]", shape.name(), f.summary(),
											startPos, w, win[2], b, i)
									.isEqualTo(Float.floatToRawIntBits(fp16Out[b][i]));
				}
			}
		}
	}

	@Test
	@DisplayName("the quantized kernel refuses a head width that is not a multiple of 32, as DeviceKvFormat does")
	void quantizedLaunch_refusesAHeadWidthNotAMultipleOf32() {
		// kvDim 288 is whole blocks, so the formats store it; a head of 36 puts heads across block boundaries, which
		// the decode lanes and the eight-value prefill reads do not handle.
		int headDim = 36;
		int numHeads = 8;
		int kvDim = 8 * headDim;
		int batch = 4;
		Random rng = new Random(5);
		GpuBindings gpu = ctx.bindings();
		try (DeviceKvCache cache = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, Q8);
				Arena arena = Arena.ofConfined()) {
			cache.appendWindow(0, gaussianRows(rng, batch, kvDim), gaussianRows(rng, batch, kvDim), batch);
			long qBytes = (long) batch * numHeads * headDim * Float.BYTES;
			long ptrBytes = batch * ADDRESS.byteSize();
			MemorySegment dQ = gpu.deviceMalloc(ctx.deviceIndex(), qBytes);
			MemorySegment dOut = gpu.deviceMalloc(ctx.deviceIndex(), qBytes);
			MemorySegment dTables = gpu.deviceMalloc(ctx.deviceIndex(), 2 * ptrBytes + batch * Integer.BYTES);
			try {
				MemorySegment ht = arena.allocate(2 * ptrBytes + batch * Integer.BYTES, 8);
				for (int b = 0; b < batch; b++) {
					ht.setAtIndex(ADDRESS, b, cache.kPointer());
					ht.setAtIndex(ADDRESS, batch + b, cache.vPointer());
					ht.set(JAVA_INT, 2 * ptrBytes + (long) b * Integer.BYTES, b + 1);
				}
				DeviceStaging.copy(gpu, dTables, ht, ht.byteSize(), GpuBindings.H2D, 0, "test tables");
				for (int rowsPerBlock : new int[] { 1, 4 })
					assertThatThrownBy(() -> GqaAttentionKvKernel.tryLoad().launch(dQ, dTables, dTables.asSlice(ptrBytes),
							dTables.asSlice(2 * ptrBytes), dOut, batch, numHeads, 1, headDim, kvDim, rowsPerBlock, 0, Q8,
							null)).as("rowsPerBlock " + rowsPerBlock).isInstanceOf(IllegalArgumentException.class)
							.hasMessageContaining("multiple of 32");
			} finally {
				download(gpu, dOut, 1, arena); // waits for anything launched on the default stream
				gpu.deviceFree(dQ);
				gpu.deviceFree(dOut);
				gpu.deviceFree(dTables);
			}
		}
	}

	// ---- helpers --------------------------------------------------------------------------------------------------

	/** A prefill window of {@code q.length} rows from {@code startPos} over one cache of {@code f}, on the GPU. */
	private float[][] attendWindow(Shape shape, DeviceKvFormats f, float[][] k, float[][] v, float[][] q, int startPos,
			int window) {
		int w = q.length;
		try (DeviceKvCache cache = new DeviceKvCache(ctx, shape.kvDim(), DeviceKvCache.INITIAL_SEQ_CAPACITY, f)) {
			cache.appendWindow(0, k, v, k.length);
			int[] seqLens = new int[w];
			DeviceKvCache[] perRow = new DeviceKvCache[w];
			for (int b = 0; b < w; b++) {
				seqLens[b] = startPos + b + 1;
				perRow[b] = cache;
			}
			float[][] out = new float[w][];
			boolean ok = CudaGqaAttention.tryCreate(ctx).attendBatched(perRow, q, seqLens, out, shape.numHeads(),
					shape.headDim(), shape.gqaRatio(), shape.kvDim(), window);
			assertThat(ok).as("kernel must load").isTrue();
			return out;
		}
	}

	/**
	 * Rows whose every 32-value block is {@code q * 2^-e} for integers the host encoder of {@code format} gives back
	 * unchanged: for q8_0 one value per block at +-127 (so the scale is exactly 2^-e) and the rest in [-127, 127];
	 * for q4_0 one value at -8 (the block's first largest magnitude, scale 2^-e) and the rest in [-7, 7]. FP16 holds
	 * each value exactly (at most 8 significant bits), so an FP16 cache stores the same floats. An FP16 format gets
	 * the q8_0 construction.
	 */
	private static float[][] exactRows(Random rng, int rows, int kvDim, DeviceKvFormat format) {
		boolean q4 = format == DeviceKvFormat.Q4_0;
		float[][] out = new float[rows][kvDim];
		for (int r = 0; r < rows; r++) {
			for (int b = 0; b < kvDim; b += 32) {
				float scale = Math.scalb(1f, -(3 + rng.nextInt(6)));
				int peak = rng.nextInt(32);
				for (int i = 0; i < 32; i++) {
					int qv = q4 ? rng.nextInt(15) - 7 : rng.nextInt(255) - 127;
					if (i == peak)
						qv = q4 ? -8 : (rng.nextBoolean() ? 127 : -127);
					out[r][b + i] = qv * scale;
				}
			}
		}
		return out;
	}

	private void runStreams(Shape shape, DeviceKvFormats f, int[] positions, int window) {
		int kvDim = shape.kvDim();
		Random rng = new Random(shape.name().hashCode() * 31 + f.hashCode());
		int batch = positions.length;
		DeviceKvCache[] kv = new DeviceKvCache[batch];
		float[][] q = new float[batch][];
		int[] seqLens = new int[batch];
		float[][] kHost = new float[batch][];
		float[][] vHost = new float[batch][];
		try {
			for (int b = 0; b < batch; b++) {
				seqLens[b] = positions[b] + 1;
				float[][] k = gaussianRows(rng, seqLens[b], kvDim);
				float[][] v = gaussianRows(rng, seqLens[b], kvDim);
				kv[b] = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f);
				kv[b].appendWindow(0, k, v, seqLens[b]);
				kHost[b] = hostValues(f.k(), k, kvDim);
				vHost[b] = hostValues(f.v(), v, kvDim);
				q[b] = gaussianRow(rng, shape.numHeads() * shape.headDim());
			}
			float[][] gpuOut = new float[batch][];
			boolean ok = CudaGqaAttention.tryCreate(ctx).attendBatched(kv, q, seqLens, gpuOut, shape.numHeads(),
					shape.headDim(), shape.gqaRatio(), kvDim, window);
			assertThat(ok).as("kernel must load").isTrue();
			for (int b = 0; b < batch; b++) {
				float[] cpu = new float[shape.numHeads() * shape.headDim()];
				GqaMath.attend(q[b], kHost[b], vHost[b], seqLens[b], cpu, new float[seqLens[b]], shape.numHeads(),
						shape.headDim(), shape.gqaRatio(), kvDim, window);
				assertClose(shape.name() + " " + f.summary() + " stream " + b + " window " + window, gpuOut[b], cpu,
						seqLens[b] >= 4096 ? TOL_LONG : TOL);
			}
		} finally {
			for (DeviceKvCache c : kv)
				if (c != null)
					c.close();
		}
	}

	private void runWindow(Shape shape, DeviceKvFormats f, int startPos, int w, int window) {
		int kvDim = shape.kvDim();
		Random rng = new Random(shape.name().hashCode() * 17 + f.hashCode());
		int rows = startPos + w;
		float[][] k = gaussianRows(rng, rows, kvDim);
		float[][] v = gaussianRows(rng, rows, kvDim);
		float[] kHost = hostValues(f.k(), k, kvDim);
		float[] vHost = hostValues(f.v(), v, kvDim);
		try (DeviceKvCache cache = new DeviceKvCache(ctx, kvDim, DeviceKvCache.INITIAL_SEQ_CAPACITY, f)) {
			cache.appendWindow(0, k, v, rows);
			float[][] q = gaussianRows(rng, w, shape.numHeads() * shape.headDim());
			int[] seqLens = new int[w];
			DeviceKvCache[] perRow = new DeviceKvCache[w];
			for (int b = 0; b < w; b++) {
				seqLens[b] = startPos + b + 1;
				perRow[b] = cache;
			}
			float[][] gpuOut = new float[w][];
			boolean ok = CudaGqaAttention.tryCreate(ctx).attendBatched(perRow, q, seqLens, gpuOut, shape.numHeads(),
					shape.headDim(), shape.gqaRatio(), kvDim, window);
			assertThat(ok).as("kernel must load").isTrue();
			for (int b = 0; b < w; b++) {
				float[] cpu = new float[shape.numHeads() * shape.headDim()];
				GqaMath.attend(q[b], kHost, vHost, seqLens[b], cpu, new float[seqLens[b]], shape.numHeads(),
						shape.headDim(), shape.gqaRatio(), kvDim, window);
				assertClose(shape.name() + " " + f.summary() + " row " + b + " window " + window, gpuOut[b], cpu, TOL);
			}
		}
	}

	/** The host codec's bytes for rows of {@code format} (FP16: Float.floatToFloat16, little-endian). */
	private static byte[] hostBytes(DeviceKvFormat format, float[][] rows, int kvDim) {
		int rowBytes = (int) format.rowBytes(kvDim);
		byte[] out = new byte[rows.length * rowBytes];
		for (int r = 0; r < rows.length; r++) {
			if (format.quantized()) {
				KvRowCodec.encode(format.hostType(), rows[r], 0, kvDim, out, r * rowBytes);
			} else {
				for (int i = 0; i < kvDim; i++) {
					short h = Float.floatToFloat16(rows[r][i]);
					out[r * rowBytes + 2 * i] = (byte) h;
					out[r * rowBytes + 2 * i + 1] = (byte) (h >> 8);
				}
			}
		}
		return out;
	}

	/** The values the host holds for rows of {@code format}, row-major: the host codec's decode. */
	private static float[] hostValues(DeviceKvFormat format, float[][] rows, int kvDim) {
		float[] out = new float[rows.length * kvDim];
		byte[] enc = new byte[(int) format.rowBytes(kvDim)];
		for (int r = 0; r < rows.length; r++) {
			if (format.quantized()) {
				KvRowCodec.encode(format.hostType(), rows[r], 0, kvDim, enc, 0);
				KvRowCodec.decode(format.hostType(), enc, 0, out, r * kvDim, kvDim);
			} else {
				for (int i = 0; i < kvDim; i++)
					out[r * kvDim + i] = Float.float16ToFloat(Float.floatToFloat16(rows[r][i]));
			}
		}
		return out;
	}

	/**
	 * Rows whose 32-value blocks hit the encoders' edge cases: Gaussian values; a block whose scale is exactly 1 with
	 * values on the rounding ties (q8_0: +-0.5, +-2.5; q4_0: +-0.5, 6.5, 7); a block whose largest value appears
	 * with both signs, each sign first in turn (q4_0 keeps the first); an all-zero block; a block of 1e-6 values (FP16 scale rounds to zero);
	 * an outlier block.
	 */
	private static float[][] edgeRows(int rows, int kvDim, long seed) {
		Random rng = new Random(seed);
		float[][] out = gaussianRows(rng, rows, kvDim);
		for (int r = 0; r < rows; r++) {
			float[] row = out[r];
			int b = (r % (kvDim / 32)) * 32;
			switch (r % 6) {
			case 0 -> {
				row[b] = 127f; // q8_0: d = 1
				row[b + 1] = 2.5f;
				row[b + 2] = -2.5f;
				row[b + 3] = 0.5f;
				row[b + 4] = -0.5f;
			}
			case 1 -> {
				row[b] = -8f; // q4_0: d = 1
				row[b + 1] = -0.5f;
				row[b + 2] = 0.5f;
				row[b + 17] = 6.5f;
				row[b + 18] = 7f;
				for (int i = 3; i < 32; i++)
					if (i != 17 && i != 18)
						row[b + i] = Math.max(-7.9f, Math.min(6.9f, row[b + i]));
			}
			case 2 -> {
				// The block's largest magnitude with both signs, in both orders (q4_0 keeps the first).
				float first = (r / 6) % 2 == 0 ? 4f : -4f;
				row[b + 2] = first;
				row[b + 9] = -first;
			}
			case 3 -> java.util.Arrays.fill(row, b, b + 32, 0f);
			case 4 -> {
				for (int i = 0; i < 32; i++)
					row[b + i] = (i % 2 == 0 ? 1e-6f : -1e-6f) * (i + 1);
			}
			default -> row[b + 7] = 1e4f;
			}
		}
		return out;
	}

	private static float[][] gaussianRows(Random rng, int rows, int n) {
		float[][] out = new float[rows][];
		for (int r = 0; r < rows; r++)
			out[r] = gaussianRow(rng, n);
		return out;
	}

	private static float[] gaussianRow(Random rng, int n) {
		float[] row = new float[n];
		for (int i = 0; i < n; i++)
			row[i] = (float) rng.nextGaussian();
		return row;
	}

	private static float[][] slice(float[][] src, int from, int to) {
		float[][] out = new float[to - from][];
		System.arraycopy(src, from, out, 0, to - from);
		return out;
	}

	private static float[] flatten(float[][] rows) {
		int n = rows[0].length;
		float[] out = new float[rows.length * n];
		for (int r = 0; r < rows.length; r++)
			System.arraycopy(rows[r], 0, out, r * n, n);
		return out;
	}

	private static byte[] rowOf(byte[] all, long rowBytes, int row) {
		byte[] out = new byte[(int) rowBytes];
		System.arraycopy(all, (int) (row * rowBytes), out, 0, (int) rowBytes);
		return out;
	}

	private static float[] download(GpuBindings gpu, MemorySegment d, int n, Arena arena) {
		MemorySegment h = arena.allocate((long) n * Float.BYTES);
		DeviceStaging.copy(gpu, h, d, h.byteSize(), GpuBindings.D2H, 0, "test out");
		return h.toArray(JAVA_FLOAT);
	}

	private static void assertClose(String label, float[] actual, float[] expected, float tol) {
		assertThat(actual).as(label).hasSameSizeAs(expected);
		for (int i = 0; i < expected.length; i++)
			assertThat(actual[i]).as(label + " [%d]", i).isCloseTo(expected[i], within(tol));
	}
}
