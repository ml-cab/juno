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

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

/**
 * The RoPE rotation reads its angles from {@link RopeTable} instead of computing
 * them per head and per layer. That is only acceptable if nothing changes: every
 * case here compares against a verbatim copy of the rotation as it was before
 * the table, bit for bit.
 */
class RopeTableTest {

	private static final int[] HEAD_DIMS = { 64, 96, 128 };
	private static final float[] BASES = { 10_000f, 500_000f, 1_000_000f };

	/** Verbatim copy of the adjacent rotation before the table existed. */
	private static void ropeBefore(float[] x, int pos, int nHeads, int headDim, float ropeTheta) {
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < headDim / 2; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) Math.cos(angle);
				float sinA = (float) Math.sin(angle);
				float x0 = x[base + 2 * i];
				float x1 = x[base + 2 * i + 1];
				x[base + 2 * i] = x0 * cosA - x1 * sinA;
				x[base + 2 * i + 1] = x0 * sinA + x1 * cosA;
			}
		}
	}

	/** Verbatim copy of the split-half rotation before the table existed. */
	private static void ropeSplitBefore(float[] x, int pos, int nHeads, int headDim, float ropeTheta) {
		int half = headDim / 2;
		for (int h = 0; h < nHeads; h++) {
			int base = h * headDim;
			for (int i = 0; i < half; i++) {
				double freq = 1.0 / Math.pow(ropeTheta, (2.0 * i) / headDim);
				double angle = pos * freq;
				float cosA = (float) Math.cos(angle);
				float sinA = (float) Math.sin(angle);
				float x0 = x[base + i];
				float x1 = x[base + i + half];
				x[base + i] = x0 * cosA - x1 * sinA;
				x[base + i + half] = x0 * sinA + x1 * cosA;
			}
		}
	}

	@Test
	void adjacent_rotation_is_bit_identical_to_the_rotation_before_the_table() {
		Random rng = new Random(42);
		for (int headDim : HEAD_DIMS) {
			for (float theta : BASES) {
				for (int k = 0; k < 200; k++) {
					int pos = k < 3 ? new int[] { 0, 1, RopeTable.MAX_POSITIONS - 1 }[k] : rng.nextInt(RopeTable.MAX_POSITIONS);
					float[] x = random(3 * headDim, rng);
					float[] before = x.clone();
					ropeBefore(before, pos, 3, headDim, theta);
					LlamaTransformerHandler.rope(x, pos, 3, headDim, theta);
					assertThat(x).as("headDim %d theta %s pos %d", headDim, theta, pos).containsExactly(before);
				}
			}
		}
	}

	@Test
	void split_half_rotation_is_bit_identical_to_the_rotation_before_the_table() {
		Random rng = new Random(43);
		for (int headDim : HEAD_DIMS) {
			for (float theta : BASES) {
				for (int k = 0; k < 200; k++) {
					int pos = rng.nextInt(RopeTable.MAX_POSITIONS);
					float[] x = random(3 * headDim, rng);
					float[] before = x.clone();
					ropeSplitBefore(before, pos, 3, headDim, theta);
					LlamaTransformerHandler.rope(x, pos, 3, headDim, theta, RopePairing.SPLIT_HALF);
					assertThat(x).as("headDim %d theta %s pos %d", headDim, theta, pos).containsExactly(before);
				}
			}
		}
	}

	@Test
	void positions_past_the_table_are_computed_in_place_and_still_identical() {
		Random rng = new Random(44);
		for (int pos : new int[] { RopeTable.MAX_POSITIONS, RopeTable.MAX_POSITIONS + 1, 100_000 }) {
			float[] x = random(2 * 128, rng);
			float[] before = x.clone();
			ropeBefore(before, pos, 2, 128, 1_000_000f);
			LlamaTransformerHandler.rope(x, pos, 2, 128, 1_000_000f);
			assertThat(x).as("pos %d", pos).containsExactly(before);
		}
		assertThat(RopeTable.of(128, 1_000_000f).block(RopeTable.MAX_POSITIONS)).isNull();
	}

	@Test
	void one_table_per_head_size_and_base() {
		assertThat(RopeTable.of(64, 10_000f)).isSameAs(RopeTable.of(64, 10_000f));
		assertThat(RopeTable.of(64, 10_000f)).isNotSameAs(RopeTable.of(128, 10_000f));
		assertThat(RopeTable.of(64, 10_000f)).isNotSameAs(RopeTable.of(64, 500_000f));
	}

	@Test
	void concurrent_readers_of_cold_blocks_all_see_identical_values() throws Exception {
		// A base no other test uses, so every block starts cold and threads race to
		// publish it.
		float theta = 123_457f;
		int headDim = 128;
		int threads = 8;
		ExecutorService pool = Executors.newFixedThreadPool(threads);
		CountDownLatch start = new CountDownLatch(1);
		List<Future<Integer>> results = new ArrayList<>();
		for (int t = 0; t < threads; t++) {
			long seed = t;
			results.add(pool.submit(() -> {
				start.await();
				Random rng = new Random(seed);
				int mismatches = 0;
				for (int k = 0; k < 2000; k++) {
					int pos = rng.nextInt(RopeTable.MAX_POSITIONS);
					float[] x = random(2 * headDim, rng);
					float[] before = x.clone();
					ropeBefore(before, pos, 2, headDim, theta);
					LlamaTransformerHandler.rope(x, pos, 2, headDim, theta);
					if (!java.util.Arrays.equals(x, before))
						mismatches++;
				}
				return mismatches;
			}));
		}
		start.countDown();
		for (Future<Integer> f : results)
			assertThat(f.get(60, TimeUnit.SECONDS)).isZero();
		pool.shutdown();
	}

	@Test
	void a_warm_rotation_allocates_nothing() {
		com.sun.management.ThreadMXBean mx = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		float[] x = random(32 * 64, new Random(45));
		// Warm the table blocks and the JIT for the positions measured below.
		for (int r = 0; r < 20_000; r++) {
			LlamaTransformerHandler.rope(x, r % 512, 32, 64, 10_000f);
			LlamaTransformerHandler.rope(x, r % 512, 32, 64, 10_000f, RopePairing.SPLIT_HALF);
		}
		long tid = Thread.currentThread().threadId();
		long before = mx.getThreadAllocatedBytes(tid);
		for (int r = 0; r < 10_000; r++) {
			LlamaTransformerHandler.rope(x, r % 512, 32, 64, 10_000f);
			LlamaTransformerHandler.rope(x, r % 512, 32, 64, 10_000f, RopePairing.SPLIT_HALF);
		}
		long allocated = mx.getThreadAllocatedBytes(tid) - before;
		// 20000 calls; any per-call allocation would be at least 16 bytes each.
		assertThat(allocated).as("bytes allocated by 20000 warm rotations").isLessThan(20_000L);
	}

	private static float[] random(int n, Random rng) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) rng.nextGaussian();
		return a;
	}
}
