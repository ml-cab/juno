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

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The KV mirror's window append: a prefill window's K and V rows written as one
 * contiguous copy each (from host rows) or cast in place on the device (from the
 * prefill-window region's device rows), instead of one copy per row per token.
 * The mirror must end up holding exactly what the per-token append writes, its
 * watermark must cover the window only once the window is complete and abuts the
 * written prefix, and growth across a window must keep the prefix.
 */
@Tag("gpu")
@DisplayName("DeviceKvCache - prefill window append")
class DeviceKvWindowAppendTest {

	private static final int KV_DIM = 256;

	private static GpuContext ctx;

	@TempDir
	Path tmp;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(PrefillWindowKernels.tryLoad() != null, "prefill-window kernels failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a host window append holds what the per-token append holds, and covers the window")
	void hostWindow_matchesPerTokenAppend() {
		float[][] k = rows(100, 1);
		float[][] v = rows(100, 2);
		try (DeviceKvCache perToken = new DeviceKvCache(ctx, KV_DIM);
				DeviceKvCache window = new DeviceKvCache(ctx, KV_DIM)) {
			for (int p = 0; p < 100; p++)
				perToken.appendToken(p, k[p], v[p]);
			window.appendWindow(0, slice(k, 0, 37), slice(v, 0, 37), 37);
			window.appendWindow(37, slice(k, 37, 100), slice(v, 37, 100), 63);

			assertThat(window.validTokens()).isEqualTo(100);
			assertThat(window.downloadK(100)).containsExactly(perToken.downloadK(100));
			assertThat(window.downloadV(100)).containsExactly(perToken.downloadV(100));
		}
	}

	@Test
	@DisplayName("a host window append is one K copy and one V copy, not one per row")
	void hostWindow_isOneCopyPerTensor() throws Exception {
		float[][] k = rows(512, 3);
		float[][] v = rows(512, 4);
		try (DeviceKvCache cache = new DeviceKvCache(ctx, KV_DIM)) {
			List<RecordedEvent> events = record(() -> cache.appendWindow(0, k, v, 512));
			List<RecordedEvent> h2d = events.stream()
					.filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
					.filter(e -> "H2D".equals(e.getString("direction"))).toList();
			long copies = h2d.stream().mapToLong(e -> e.getLong("copies")).sum();
			long bytes = h2d.stream().mapToLong(e -> e.getLong("bytes")).sum();
			assertThat(copies).as("host-to-device copies for a 512-row window").isEqualTo(2);
			assertThat(bytes).isEqualTo(2L * 512 * KV_DIM * Short.BYTES);
		}
	}

	@Test
	@DisplayName("a device window write casts in place: no host copy, same bits, watermark only once marked")
	void deviceWindow_castsInPlaceAndWaitsForTheMark() throws Exception {
		float[][] k = rows(64, 5);
		float[][] v = rows(64, 6);
		PrefillWindowKernels kernels = PrefillWindowKernels.tryLoad();
		try (DeviceKvCache reference = new DeviceKvCache(ctx, KV_DIM);
				DeviceKvCache cache = new DeviceKvCache(ctx, KV_DIM);
				ResidentChain chain = ResidentChain.open(ctx)) {
			reference.appendWindow(0, k, v, 64);
			ResidentActivation dk = chain.allocate(64, KV_DIM);
			ResidentActivation dv = chain.allocate(64, KV_DIM);
			dk.upload(k);
			dv.upload(v);
			chain.sync();

			List<RecordedEvent> events = record(() -> {
				cache.writeWindowOnDevice(0, 64, dk.devicePointer(), dv.devicePointer(), kernels, chain.stream());
				chain.sync();
			});
			long h2d = events.stream().filter(e -> e.getEventType().getName().equals("juno.DeviceStaging"))
					.filter(e -> "H2D".equals(e.getString("direction"))).mapToLong(e -> e.getLong("copies")).sum();
			assertThat(h2d).as("host-to-device copies for a device window write").isZero();

			assertThat(cache.validTokens()).as("watermark before the window is marked").isZero();
			assertThat(cache.readableThrough(1)).isFalse();
			cache.markWritten(0, 64);
			assertThat(cache.validTokens()).isEqualTo(64);
			assertThat(cache.downloadK(64)).containsExactly(reference.downloadK(64));
			assertThat(cache.downloadV(64)).containsExactly(reference.downloadV(64));
		}
	}

	@Test
	@DisplayName("a window that crosses the capacity grows the mirror and keeps the prefix")
	void windowAcrossGrowth_keepsThePrefix() {
		float[][] k = rows(300, 7);
		float[][] v = rows(300, 8);
		try (DeviceKvCache perToken = new DeviceKvCache(ctx, KV_DIM);
				DeviceKvCache window = new DeviceKvCache(ctx, KV_DIM)) {
			for (int p = 0; p < 300; p++)
				perToken.appendToken(p, k[p], v[p]);
			window.appendWindow(0, slice(k, 0, 40), slice(v, 0, 40), 40);
			assertThat(window.capacityTokens()).isEqualTo(DeviceKvCache.INITIAL_SEQ_CAPACITY);
			window.appendWindow(40, slice(k, 40, 300), slice(v, 40, 300), 260);
			assertThat(window.capacityTokens()).isGreaterThanOrEqualTo(300);
			assertThat(window.downloadK(300)).containsExactly(perToken.downloadK(300));
			assertThat(window.downloadV(300)).containsExactly(perToken.downloadV(300));
		}
	}

	@Test
	@DisplayName("a window that does not abut the written prefix is stored but not readable")
	void windowPastThePrefix_isNotReadable() {
		try (DeviceKvCache cache = new DeviceKvCache(ctx, KV_DIM)) {
			cache.appendWindow(10, rows(5, 9), rows(5, 10), 5);
			assertThat(cache.validTokens()).isZero();
			assertThat(cache.readableThrough(15)).isFalse();
			cache.appendWindow(0, rows(10, 11), rows(10, 12), 10);
			assertThat(cache.validTokens()).as("the gap is filled, but the later rows are not re-validated").isEqualTo(10);
		}
	}

	@Test
	@DisplayName("a window that cannot grow the mirror fails with a device-memory error the callers recognize")
	void windowGrowthOutOfMemory_isAVramOom() {
		GpuBindings gpu = ctx.bindings();
		try (DeviceKvCache cache = new DeviceKvCache(ctx, KV_DIM)) {
			cache.appendWindow(0, rows(64, 13), rows(64, 14), 64);
			MemorySegment ballast = fillDevice(gpu, 4L << 20);
			try {
				// 8192 positions of 256 FP16 values per tensor need 4 MiB each, more than is left.
				assertThatThrownBy(() -> cache.appendWindow(64, rows(8128, 15), rows(8128, 16), 8128))
						.isInstanceOf(IllegalStateException.class)
						.matches(ex -> GpuLayerOffload.isVramOom((IllegalStateException) ex), "is a VRAM OOM");
				assertThat(cache.validTokens()).as("the failed window does not move the watermark").isEqualTo(64);
			} finally {
				gpu.deviceFree(ballast);
			}
		}
	}

	// ── helpers ────────────────────────────────────────────────────────────────

	/** Allocates all but about {@code leave} bytes of free device memory and returns the allocation. */
	private static MemorySegment fillDevice(GpuBindings gpu, long leave) {
		long free = gpu.memGetInfo(ctx.deviceIndex())[0];
		assumeTrue(free > leave + (64L << 20), "not enough free device memory to set up the test");
		long bytes = free - leave;
		while (true) {
			try {
				return gpu.deviceMalloc(ctx.deviceIndex(), bytes);
			} catch (IllegalStateException ex) {
				bytes -= 16L << 20; // the allocator keeps some of what memGetInfo reports for itself
				assumeTrue(bytes > 0, "could not reserve device memory");
			}
		}
	}

	private List<RecordedEvent> record(Runnable body) throws Exception {
		Path jfr = tmp.resolve("kv-" + System.nanoTime() + ".jfr");
		try (Recording rec = new Recording()) {
			rec.enable("juno.DeviceStaging").withThreshold(java.time.Duration.ZERO);
			rec.setDestination(jfr);
			rec.start();
			body.run();
			rec.stop();
		}
		return RecordingFile.readAllEvents(jfr);
	}

	private static float[][] rows(int n, long seed) {
		Random r = new Random(seed);
		float[][] out = new float[n][KV_DIM];
		for (float[] row : out)
			for (int i = 0; i < KV_DIM; i++)
				row[i] = (float) r.nextGaussian();
		return out;
	}

	private static float[][] slice(float[][] x, int from, int to) {
		float[][] out = new float[to - from][];
		System.arraycopy(x, from, out, 0, to - from);
		return out;
	}
}
