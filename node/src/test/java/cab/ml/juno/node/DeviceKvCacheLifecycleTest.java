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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Leak-freedom and grow-and-preserve correctness for {@link DeviceKvCache} — the
 * device-resident KV mirror behind the GPU-resident attention path
 * ({@code --gpu-attention}). Exercised directly (no handler wiring needed to
 * prove the plumbing itself doesn't leak or corrupt data).
 *
 * <p>Run: {@code mvn test -Dgroups=gpu -pl node -Dtest=DeviceKvCacheLifecycleTest}.
 */
@Tag("gpu")
@DisplayName("DeviceKvCache — lifecycle (no leak) and grow-and-preserve correctness")
class DeviceKvCacheLifecycleTest {

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping — no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("50 create+evict cycles (each forcing growth) return allocated bytes to baseline")
	void create_and_close_50_requests_leaves_no_leak() {
		long baseline = DeviceKvCache.allocatedBytes();
		int kvDim = 16;
		Random rng = new Random(42);

		for (int req = 0; req < 50; req++) {
			DeviceKvCache[] layers = DeviceKvCache.newLayers(ctx, 3, kvDim);
			try {
				// Force at least one capacity growth per request (initial=64).
				for (DeviceKvCache layer : layers) {
					for (int pos = 0; pos < 130; pos++) {
						layer.appendToken(pos, randomRow(rng, kvDim), randomRow(rng, kvDim));
					}
				}
			} finally {
				for (DeviceKvCache layer : layers)
					layer.close();
			}
		}

		assertThat(DeviceKvCache.allocatedBytes())
				.as("allocated bytes must return to baseline after every request is closed")
				.isEqualTo(baseline);
	}

	@Test
	@DisplayName("grow-and-preserve keeps rows written before growth intact")
	void grow_preserves_previously_written_rows() {
		int kvDim = 8;
		Random rng = new Random(7);
		DeviceKvCache kv = new DeviceKvCache(ctx, kvDim, 4); // tiny initial capacity, guarantees growth

		float[][] kRows = new float[20][];
		float[][] vRows = new float[20][];
		try {
			for (int pos = 0; pos < 20; pos++) {
				kRows[pos] = randomRow(rng, kvDim);
				vRows[pos] = randomRow(rng, kvDim);
				kv.appendToken(pos, kRows[pos], vRows[pos]); // crosses the 4 -> 8 -> 16 -> 32 growth boundaries
			}

			float[] kBack = kv.downloadK(20);
			float[] vBack = kv.downloadV(20);
			for (int pos = 0; pos < 20; pos++) {
				for (int d = 0; d < kvDim; d++) {
					// FP16 storage — tolerance covers rounding, not data loss/corruption.
					assertThat(kBack[pos * kvDim + d]).as("K[%d][%d]", pos, d)
							.isCloseTo(kRows[pos][d], within(2e-3f));
					assertThat(vBack[pos * kvDim + d]).as("V[%d][%d]", pos, d)
							.isCloseTo(vRows[pos][d], within(2e-3f));
				}
			}
		} finally {
			kv.close();
		}
	}

	/**
	 * Device memory leaked by a failed allocation is gigabytes here, and the desktop's
	 * own use of the card moves the free figure by tens of MiB between two readings.
	 */
	private static final long FREE_TOLERANCE_BYTES = 256L * 1024 * 1024;

	/** kvDim at which one K (or V) tensor at the initial capacity is {@code bytes} long. */
	private static int kvDimForTensorBytes(long bytes) {
		return (int) (bytes / ((long) DeviceKvCache.INITIAL_SEQ_CAPACITY * Short.BYTES));
	}

	@Test
	@DisplayName("A request's mirrors that run out of memory part-way are all released")
	void a_partly_allocated_set_of_layers_is_released_when_one_runs_out() {
		// On a card at its capacity, allocating a request's mirrors layer by layer can
		// succeed for the first layers and fail on a later one. The request then runs
		// attention on the CPU; the layers already allocated must not stay behind, or
		// every such request takes device memory for good.
		long baselineBytes = DeviceKvCache.allocatedBytes();
		long freeBefore = ctx.freeVramBytes();
		int kvDim = kvDimForTensorBytes(freeBefore / 12); // each layer about a sixth of free memory
		IllegalStateException oom = org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
				() -> DeviceKvCache.newLayers(ctx, 12, kvDim));
		assertThat(GpuLayerOffload.isVramOom(oom)).as("the failure is the allocator's").isTrue();
		assertThat(DeviceKvCache.allocatedBytes()).as("no mirror bytes left counted").isEqualTo(baselineBytes);
		assertThat(ctx.freeVramBytes()).as("device memory given back")
				.isGreaterThan(freeBefore - FREE_TOLERANCE_BYTES);
	}

	@Test
	@DisplayName("A mirror whose V tensor does not fit gives its K tensor back")
	void a_mirror_whose_second_tensor_does_not_fit_releases_the_first() {
		long freeBefore = ctx.freeVramBytes();
		int kvDim = kvDimForTensorBytes(freeBefore * 6 / 10); // K fits, K and V do not
		IllegalStateException oom = org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
				() -> new DeviceKvCache(ctx, kvDim));
		assertThat(GpuLayerOffload.isVramOom(oom)).as("the failure is the allocator's").isTrue();
		assertThat(ctx.freeVramBytes()).as("device memory given back")
				.isGreaterThan(freeBefore - FREE_TOLERANCE_BYTES);
	}

	private static float[] randomRow(Random rng, int kvDim) {
		float[] row = new float[kvDim];
		for (int i = 0; i < kvDim; i++)
			row[i] = rng.nextFloat() * 2f - 1f;
		return row;
	}
}
