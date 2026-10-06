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

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The decode region on a card with no device memory left. A region's buffers are
 * allocated on first use, so a card that filled after the path was built (three
 * tensor-parallel nodes sharing one device, a partially offloaded model) runs out
 * there. That must not fail the request: the call returns {@code null}, which every
 * handler already reads as "keep the existing path", and the path stops opening new
 * regions. Fills the real device to get a real allocation failure.
 *
 * <p>Takes the whole device for a moment. Run on an otherwise idle one.
 */
@Tag("gpu")
@DisplayName("Resident decode region - out of device memory when opening a region")
class ResidentQkvPathDeviceFullTest {

	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 64;
	private static final int H = HEADS * HEAD_DIM;
	private static final int KV = KV_HEADS * HEAD_DIM;

	private static GpuContext ctx;
	private static DeviceQ4KMatrix[] wq;
	private static DeviceQ4KMatrix[] wk;
	private static DeviceQ4KMatrix[] wv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "K-quant MMQ kernel failed to load");
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
		Random rnd = new Random(3);
		wq = new DeviceQ4KMatrix[] { q4k(H, H, rnd) };
		wk = new DeviceQ4KMatrix[] { q4k(KV, H, rnd) };
		wv = new DeviceQ4KMatrix[] { q4k(KV, H, rnd) };
	}

	@AfterAll
	static void destroy() {
		for (DeviceQ4KMatrix[] a : new DeviceQ4KMatrix[][] { wq, wk, wv })
			if (a != null)
				for (DeviceQ4KMatrix m : a)
					m.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("a region that cannot be allocated gives the existing path, not a failed request, and is not retried")
	void aFullDeviceGivesTheExistingPath() {
		LlamaConfig cfg = new LlamaConfig(H, 1, HEADS, KV_HEADS, HEAD_DIM, 4 * H, 1000, 1e-5f, 10_000f, "llama");
		float[][] attnNorm = { new float[H] };
		java.util.Arrays.fill(attnNorm[0], 1f);
		float[] x = new float[H];
		java.util.Arrays.fill(x, 0.5f);
		try (ResidentQkvPath path = ResidentQkvPath.create(ctx, cfg, attnNorm, wq, wk, wv)) {
			long weightsOnly = path.deviceBytes();
			List<MemorySegment> fill = fillDevice();
			try {
				assertThat(path.run(0, x, 3)).as("no region could be opened: the existing path").isNull();
				try (ResidentQkvPath.Lease lease = path.lease()) {
					assertThat(path.run(0, x, 3, null, lease)).as("inside a lease too").isNull();
				}
			} finally {
				for (MemorySegment m : fill)
					ctx.bindings().deviceFree(m);
			}
			assertThat(path.run(0, x, 4)).as("the path stops opening regions once the device ran out").isNull();
			assertThat(path.deviceBytes()).as("no region was opened or kept").isEqualTo(weightsOnly);
		}
	}

	/**
	 * Allocates device memory until even a 256-byte allocation fails (the runtime
	 * places small requests in pages it already holds); returns what it
	 * holds. Earlier tests in the same JVM free device memory from cleaners when their
	 * objects are collected, which can hand memory back after a fill; so collect first,
	 * fill, then collect again and top the fill up.
	 */
	private static List<MemorySegment> fillDevice() {
		List<MemorySegment> held = new ArrayList<>();
		for (int pass = 0; pass < 2; pass++) {
			System.gc();
			try {
				Thread.sleep(200);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
			fillOnce(held);
		}
		return held;
	}

	private static void fillOnce(List<MemorySegment> held) {
		for (long chunk = 256L << 20; chunk >= 256; chunk >>= 2) {
			while (true) {
				try {
					held.add(ctx.bindings().deviceMalloc(ctx.deviceIndex(), chunk));
				} catch (IllegalStateException e) {
					assertThat(GpuLayerOffload.isVramOom(e)).as("filling stops on an allocation failure").isTrue();
					break;
				}
			}
		}
	}

	private static DeviceQ4KMatrix q4k(int rows, int cols, Random rnd) {
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = rnd.nextFloat() * 2f - 1f;
		return DeviceQ4KMatrix.upload(ctx, GgufKQuantCodec.encode(host, QuantizationLayout.TYPE_Q4_K), rows, cols);
	}
}
