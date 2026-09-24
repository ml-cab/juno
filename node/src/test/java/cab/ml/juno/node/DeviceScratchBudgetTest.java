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

import org.junit.jupiter.api.Test;

/**
 * Sizing the device memory that must stay free for inference, so filling VRAM
 * with weights does not leave the first prefill with nowhere to put its
 * dequantized weight matrix.
 *
 * <p>The numbers below are taken from the model that exposed this: a 30B Llama
 * with hidden 6656, FFN 17920 and 60 layers on an 8 GiB card. Uploading layers
 * until the card was full loaded fine and decoded fine, then died on the first
 * prompt longer than eight tokens, because that is the batch width at which the
 * Q4_K path stops going row-by-row and dequantizes the whole weight matrix into
 * a device scratch buffer.
 */
class DeviceScratchBudgetTest {

	private static final long MIB = 1024L * 1024L;

	/** hidden 6656, kv 6656, ffn 17920 -- the file this bug was found on. */
	private static long thirtyB() {
		return DeviceScratchBudget.reserveBytes(6656, 6656, 17920);
	}

	@Test
	void reserveCoversTheLargestDequantizedWeightMatrix() {
		// The widest matmul is the FFN pair: 6656 x 17920 halves = 227.5 MiB, and
		// the scratch holds one whole dequantized matrix at a time.
		long largestMatrix = 6656L * 17920L * Short.BYTES;
		assertThat(largestMatrix).isEqualTo(238_551_040L);
		assertThat(thirtyB()).isGreaterThanOrEqualTo(largestMatrix);
	}

	@Test
	void reserveIsDrivenByTheWidestDimensionNotTheHiddenSize() {
		// A wider FFN reserves more...
		assertThat(DeviceScratchBudget.reserveBytes(4096, 4096, 28672))
				.isGreaterThan(DeviceScratchBudget.reserveBytes(4096, 4096, 11008));
		// ...and so does a KV wide enough to overtake the FFN.
		assertThat(DeviceScratchBudget.reserveBytes(4096, 16384, 11008))
				.isGreaterThan(DeviceScratchBudget.reserveBytes(4096, 1024, 11008));
		// Below that, the FFN sets the figure and the KV does not move it: the
		// scratch holds one matrix, so only the widest one matters.
		assertThat(DeviceScratchBudget.reserveBytes(4096, 8192, 11008))
				.isEqualTo(DeviceScratchBudget.reserveBytes(4096, 1024, 11008));
	}

	@Test
	void reserveLeavesHeadroomBeyondTheBareMatrix() {
		// Staging buffers and per-request activations share the same device, so a
		// reserve exactly equal to the matrix would still fail.
		long largestMatrix = 6656L * 17920L * Short.BYTES;
		assertThat(thirtyB()).isGreaterThan(largestMatrix);
	}

	@Test
	void reserveStaysAPlausibleFractionOfASmallCard() {
		// It has to fit on the 8 GiB card this was found on, with room for weights:
		// a reserve that swallows the card would offload nothing at all.
		assertThat(thirtyB()).isLessThan(1024 * MIB);
	}

	@Test
	void smallModelsReserveProportionallyLess() {
		long tiny = DeviceScratchBudget.reserveBytes(2048, 256, 5632);   // TinyLlama shape
		assertThat(tiny).isLessThan(thirtyB());
		assertThat(tiny).isGreaterThan(0L);
	}

	@Test
	void invalidDimensionsAreRejectedRatherThanSilentlyReservingNothing() {
		org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
				() -> DeviceScratchBudget.reserveBytes(0, 256, 5632));
		org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
				() -> DeviceScratchBudget.reserveBytes(2048, 0, 5632));
		org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
				() -> DeviceScratchBudget.reserveBytes(2048, 256, 0));
	}

	// ── the GPU-attention KV mirror ──────────────────────────────────────────

	@Test
	void kvMirrorReserveCoversKAndVForEveryLayer() {
		// One K and one V buffer per layer, each holding initialTokens x kvDim halves.
		long perLayer = 2L * 64L * 6656L * Short.BYTES;
		assertThat(DeviceScratchBudget.kvMirrorBytes(20, 6656, 64)).isEqualTo(20 * perLayer);
	}

	@Test
	void kvMirrorReserveScalesWithLayersAndContext() {
		assertThat(DeviceScratchBudget.kvMirrorBytes(40, 6656, 64))
				.isEqualTo(2 * DeviceScratchBudget.kvMirrorBytes(20, 6656, 64));
		assertThat(DeviceScratchBudget.kvMirrorBytes(20, 6656, 128))
				.isEqualTo(2 * DeviceScratchBudget.kvMirrorBytes(20, 6656, 64));
	}

	@Test
	void kvMirrorIsZeroWhenGpuAttentionIsNotInPlay() {
		// Nothing to reserve when no device mirror will be allocated; reserving anyway
		// would offload fewer weight layers for no reason.
		assertThat(DeviceScratchBudget.kvMirrorBytes(0, 6656, 64)).isZero();
	}

	@Test
	void theTwoReservesAddUpForAShardOfTheThirtyBModel() {
		// A 3-node local pipeline gives each handler 20 of the 60 layers, and all
		// three share one card: matmul scratch plus that shard's KV mirror.
		long total = thirtyB() + DeviceScratchBudget.kvMirrorBytes(20, 6656, 64);
		assertThat(total).isGreaterThan(thirtyB());
		assertThat(total).isLessThan(1024 * MIB);
	}

	// ── the upload stop rule ─────────────────────────────────────────────────

	@Test
	void uploadStopsWhileTheReserveWouldStillBeIntact() {
		long reserve = 300 * MIB;
		// 1 GiB free, a 310 MiB layer: room for the layer and the reserve after it.
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(1024 * MIB, 310 * MIB, reserve)).isTrue();
	}

	@Test
	void uploadStopsWhenTheNextLayerWouldEatTheReserve() {
		long reserve = 300 * MIB;
		// 500 MiB free: the layer fits, but it would leave only 190 MiB -- which is
		// exactly the state that used to load successfully and then crash on prefill.
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(500 * MIB, 310 * MIB, reserve)).isFalse();
	}

	@Test
	void anUnknownLayerCostDoesNotStopTheFirstUpload() {
		// Per-layer cost is learned by measuring the first upload, so before that
		// measurement exists the rule must not refuse to start.
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(1024 * MIB, 0L, 300 * MIB)).isTrue();
	}

	@Test
	void anUnreadableFreeVramFigureDoesNotStopTheUpload() {
		// memGetInfo returns 0 when the query fails; that is "unknown", and the
		// pre-existing catch-the-OOM behaviour has to stay reachable rather than
		// this rule silently disabling GPU offload altogether.
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(0L, 310 * MIB, 300 * MIB)).isTrue();
	}
}
