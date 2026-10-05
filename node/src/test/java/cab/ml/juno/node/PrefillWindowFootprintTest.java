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
 * The device bytes of one prefill-window region window, which the upload reserve
 * and the adaptive prefill chunk are both sized from. The expected figures are
 * worked out by hand from the buffers {@code PrefillWindowRegion.Window} allocates,
 * on Mistral 7B's shape (hidden 4096, 32 query heads, 8 KV heads of 128, FFN 14336).
 */
class PrefillWindowFootprintTest {

	private static final PrefillWindowRegion.Shape MISTRAL = new PrefillWindowRegion.Shape(4096, 4096, 1024, 14336,
			32, 8, 128, 4, 1e-5f);

	@Test
	void windowBuffersMatchWhatTheWindowAllocates() {
		// FP32 activations x, q, k, v, attn: 64 x (4096 + 4096 + 1024 + 1024 + 4096) x 4
		long activations = 64L * 14336 * 4;
		// FP32 scratch xn, xIn, proj (hidden each) and gateUp (2 x FFN)
		long scratch = 3L * 64 * 4096 * 4 + 64L * 2 * 14336 * 4;
		// FP16 GEMM input at the widest matmul input (the FFN), and the attention tables
		long xh = 64L * 14336 * 2;
		long tables = 64L * (2 * 8 + 4);
		assertThat(PrefillWindowFootprint.windowBytes(MISTRAL, false, 64))
				.isEqualTo(activations + scratch + xh + tables)
				.isEqualTo(15_992_064L);
	}

	@Test
	void theQ8CopyIsCountedAtTheWindowsOwnWidth() {
		// The backend's Q8_1 copy of the widest matmul input: 36 bytes per 32 values.
		assertThat(PrefillWindowFootprint.q8Bytes(MISTRAL, 64)).isEqualTo(64L * 14336 / 32 * 36);
		assertThat(PrefillWindowFootprint.bytes(MISTRAL, false, 64)).isEqualTo(15_992_064L + 1_032_192L);
	}

	@Test
	void windowBuffersAreAllocatedAtTheRoundedUpCapacity() {
		// The region rounds a window up to a multiple of 64 rows, so 9 and 64 rows cost
		// the same and 65 rows costs as much as 128.
		assertThat(PrefillWindowFootprint.capacity(9)).isEqualTo(64);
		assertThat(PrefillWindowFootprint.capacity(65)).isEqualTo(128);
		assertThat(PrefillWindowFootprint.windowBytes(MISTRAL, false, 9))
				.isEqualTo(PrefillWindowFootprint.windowBytes(MISTRAL, false, 64));
		assertThat(PrefillWindowFootprint.windowBytes(MISTRAL, false, 65))
				.isEqualTo(PrefillWindowFootprint.windowBytes(MISTRAL, false, 128));
	}

	@Test
	void aFusedQkvProjectionAddsItsActivation() {
		long fusedRow = (4096L + 2 * 1024) * Float.BYTES;
		assertThat(PrefillWindowFootprint.windowBytes(MISTRAL, true, 64)
				- PrefillWindowFootprint.windowBytes(MISTRAL, false, 64)).isEqualTo(64 * fusedRow);
	}

	@Test
	void attentionAddsNoTermThatGrowsWithTheSquareOfTheWindow() {
		// The attention kernel streams the keys and keeps no scores buffer, so a window
		// covering the start of the prompt costs linearly in its width: doubling the
		// rows doubles the footprint at most (the row buffers), never quadruples it.
		assertThat(PrefillWindowFootprint.bytes(MISTRAL, false, 1024))
				.isLessThanOrEqualTo(2 * PrefillWindowFootprint.bytes(MISTRAL, false, 512));
	}

	@Test
	void footprintNeverShrinksAsTheWindowWidens() {
		// The adaptive chunk size searches this function, so it must be monotonic.
		long previous = 0;
		for (int rows = 9; rows <= 700; rows++) {
			long b = PrefillWindowFootprint.bytes(MISTRAL, false, rows);
			assertThat(b).as("rows " + rows).isGreaterThanOrEqualTo(previous);
			previous = b;
		}
	}

	@Test
	void nonPositiveWidthsAreRejected() {
		assertThrows(IllegalArgumentException.class, () -> PrefillWindowFootprint.capacity(0));
		assertThrows(IllegalArgumentException.class, () -> PrefillWindowFootprint.bytes(MISTRAL, false, 0));
	}
}
