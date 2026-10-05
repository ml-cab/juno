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
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Sizing the device memory that must stay free for inference once the weights are
 * uploaded, so filling VRAM with weights does not leave the first prefill window
 * with nowhere to run.
 *
 * <p>The numbers below are taken from the model that exposed this: a 30B Llama
 * with hidden 6656, FFN 17920, 52 heads and 60 layers on an 8 GiB card. Uploading
 * layers until the card was full loaded fine and decoded fine, then died on the
 * first prompt longer than eight tokens, the width at which the batched path
 * starts. The batched path then dequantized a whole weight matrix into a device
 * scratch buffer; it now multiplies the packed weights directly, so what has to
 * stay free is the prefill window, and the weight-shaped term remains only for the
 * dequantizing fallback.
 */
class DeviceScratchBudgetTest {

	private static final long MIB = 1024L * 1024L;

	private static final PrefillWindowRegion.Shape THIRTY_B = new PrefillWindowRegion.Shape(6656, 6656, 6656, 17920,
			52, 52, 128, 1, 1e-5f);

	/** The 30B model's region window at the reserved width. */
	private static long thirtyBWindow() {
		return PrefillWindowFootprint.bytes(THIRTY_B, false, DeviceScratchBudget.RESERVED_WINDOW_ROWS);
	}

	private static long thirtyBDequant() {
		return DeviceScratchBudget.dequantScratchBytes(6656, 6656, 17920);
	}

	// ── the dequantizing fallback's weight-shaped term ───────────────────────

	@Test
	void dequantScratchIsTheLargestFp16WeightMatrix() {
		// The widest matmul is the FFN pair: 6656 x 17920 halves = 227.5 MiB, and the
		// dequantizing route holds one whole dequantized matrix at a time.
		assertThat(thirtyBDequant()).isEqualTo(6656L * 17920L * Short.BYTES).isEqualTo(238_551_040L);
	}

	@Test
	void dequantScratchIsDrivenByTheWidestDimensionNotTheHiddenSize() {
		assertThat(DeviceScratchBudget.dequantScratchBytes(4096, 4096, 28672))
				.isGreaterThan(DeviceScratchBudget.dequantScratchBytes(4096, 4096, 11008));
		assertThat(DeviceScratchBudget.dequantScratchBytes(4096, 16384, 11008))
				.isGreaterThan(DeviceScratchBudget.dequantScratchBytes(4096, 1024, 11008));
		// Below that, the FFN sets the figure: the scratch holds one matrix.
		assertThat(DeviceScratchBudget.dequantScratchBytes(4096, 8192, 11008))
				.isEqualTo(DeviceScratchBudget.dequantScratchBytes(4096, 1024, 11008));
	}

	@Test
	void invalidDimensionsAreRejectedRatherThanSilentlyReservingNothing() {
		assertThrows(IllegalArgumentException.class, () -> DeviceScratchBudget.dequantScratchBytes(0, 256, 5632));
		assertThrows(IllegalArgumentException.class, () -> DeviceScratchBudget.dequantScratchBytes(2048, 0, 5632));
		assertThrows(IllegalArgumentException.class, () -> DeviceScratchBudget.dequantScratchBytes(2048, 256, 0));
	}

	// ── the reserve ──────────────────────────────────────────────────────────

	@Test
	void theReservedWindowIsTheRegionsSmallestWindow() {
		// Any window wider than the host path's eight rows allocates at least this many
		// rows, so reserving fewer would guarantee nothing.
		assertThat(DeviceScratchBudget.RESERVED_WINDOW_ROWS)
				.isEqualTo(PrefillWindowFootprint.capacity(PrefillWindowRegion.MAX_HOST_WINDOW + 1));
	}

	@Test
	void reserveCoversTheWindowWithHeadroom() {
		// Host staging and the matmul library's workspace share the device, so a reserve
		// exactly equal to the window would still fail.
		long window = thirtyBWindow();
		assertThat(DeviceScratchBudget.reserveBytes(window, 0L)).isGreaterThan(window);
	}

	@Test
	void reserveIncludesWhatTheAllocatorWithholds() {
		// With the card full, the free-memory query still reports 44 to 54 MiB that no
		// allocation can obtain (GTX 1080, measured by PrefillReserveDeviceTest). The
		// upload stop rule reads that same query, so without this allowance a reserve
		// the size of a window would leave nothing allocatable.
		assertThat(DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES).isGreaterThanOrEqualTo(54 * MIB);
		assertThat(DeviceScratchBudget.reserveBytes(1L, 0L))
				.isGreaterThan(DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES);
		long window = thirtyBWindow();
		assertThat(DeviceScratchBudget.reserveBytes(window, 0L))
				.isGreaterThanOrEqualTo(window + DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES);
	}

	@Test
	void thePackedRouteNoLongerReservesAWeightMatrix() {
		// The tiled kernel multiplies the packed weights, so the reserve is window-shaped:
		// smaller than the one FP16 matrix the old reserve was built around.
		long packed = DeviceScratchBudget.reserveBytes(thirtyBWindow(), 0L);
		assertThat(packed).isLessThan(thirtyBDequant());
		assertThat(packed).isLessThan(128 * MIB);
	}

	@Test
	void theDequantizingFallbackStillReservesItsMatrix() {
		// Without the tiled kernel the batched path dequantizes, and needs both.
		long window = thirtyBWindow();
		assertThat(DeviceScratchBudget.reserveBytes(window, thirtyBDequant()))
				.isGreaterThanOrEqualTo(window + thirtyBDequant());
	}

	@Test
	void aWindowOfZeroBytesIsRejectedRatherThanReservingNothing() {
		assertThrows(IllegalArgumentException.class, () -> DeviceScratchBudget.reserveBytes(0L, 0L));
		assertThrows(IllegalArgumentException.class, () -> DeviceScratchBudget.reserveBytes(1L, -1L));
	}

	@Test
	void aLayerThatFitsOnlyUnderTheShrunkReserveIsUploaded() {
		// One 30B Q4_K_M layer is about 310 MiB. With that layer plus the shrunk reserve
		// free, the packed route uploads it; the dequantizing route's reserve does not.
		long layer = 310 * MIB;
		long packedReserve = DeviceScratchBudget.reserveBytes(thirtyBWindow(), 0L);
		long dequantReserve = DeviceScratchBudget.reserveBytes(thirtyBWindow(), thirtyBDequant());
		long free = layer + packedReserve + MIB;
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(free, layer, packedReserve)).isTrue();
		assertThat(DeviceScratchBudget.canUploadAnotherLayer(free, layer, dequantReserve)).isFalse();
	}

	@Test
	void smallModelsReserveProportionallyLess() {
		PrefillWindowRegion.Shape tiny = new PrefillWindowRegion.Shape(2048, 2048, 256, 5632, 32, 4, 64, 8, 1e-5f);
		long tinyReserve = DeviceScratchBudget.reserveBytes(PrefillWindowFootprint.bytes(tiny, false,
				DeviceScratchBudget.RESERVED_WINDOW_ROWS), 0L);
		assertThat(tinyReserve).isGreaterThan(0L).isLessThan(DeviceScratchBudget.reserveBytes(thirtyBWindow(), 0L));
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
	void theReservesAddUpForAShardOfTheThirtyBModel() {
		// A 3-node local pipeline gives each handler 20 of the 60 layers, and all three
		// share one card: the window plus that shard's KV mirror.
		long reserve = DeviceScratchBudget.reserveBytes(thirtyBWindow(), 0L);
		long total = reserve + DeviceScratchBudget.kvMirrorBytes(20, 6656, 64);
		assertThat(total).isGreaterThan(reserve).isLessThan(1024 * MIB);
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
