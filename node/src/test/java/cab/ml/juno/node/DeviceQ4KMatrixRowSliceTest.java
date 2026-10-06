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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A row range of a packed K-quant matrix as a view: what lets the decode region
 * multiply the parts of a fused tensor (Phi-3's Q/K/V, gate/up) separately. Each
 * part's product must equal that range of the fused product bit for bit, for every
 * packed type (Q6_K's rows sit in padded slots), and the view must own nothing.
 */
@Tag("gpu")
@DisplayName("DeviceQ4KMatrix.rowSlice - fused-tensor row views")
class DeviceQ4KMatrixRowSliceTest {

	private static final int COLS = 512;

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "K-quant MMQ kernel failed to load");
		mv = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("each part's product equals its rows of the fused product, for Q4_K, Q5_K and Q6_K")
	void partsMultiplyLikeTheirRowsOfTheWhole() {
		int[] parts = { 256, 64, 64 }; // a fused Q/K/V with a narrower K and V
		int rows = 384;
		for (int type : new int[] { QuantizationLayout.TYPE_Q4_K, QuantizationLayout.TYPE_Q5_K,
				QuantizationLayout.TYPE_Q6_K }) {
			Random rnd = new Random(type);
			try (DeviceQ4KMatrix fused = upload(rows, type, rnd)) {
				float[] x = randomVec(COLS, rnd);
				float[] whole = mv.sgemv(fused, x);
				int start = 0;
				for (int n : parts) {
					DeviceQ4KMatrix part = fused.rowSlice(start, n);
					assertThat(part.rows()).isEqualTo(n);
					assertThat(part.cols()).isEqualTo(COLS);
					assertThat(part.quantType()).isEqualTo(type);
					assertThat(mv.sgemv(part, x)).as("type %d rows [%d, %d)", type, start, start + n)
							.containsExactly(Arrays.copyOfRange(whole, start, start + n));
					start += n;
				}
			}
		}
	}

	@Test
	@DisplayName("a view owns nothing: closing it leaves the matrix usable, closing the matrix retires the view")
	void aViewOwnsNothing() {
		Random rnd = new Random(9);
		DeviceQ4KMatrix fused = upload(256, QuantizationLayout.TYPE_Q4_K, rnd);
		try {
			float[] x = randomVec(COLS, rnd);
			float[] before = mv.sgemv(fused, x);
			DeviceQ4KMatrix part = fused.rowSlice(0, 128);
			part.close();
			assertThat(part.isClosed()).isTrue();
			assertThat(fused.isClosed()).isFalse();
			assertThat(mv.sgemv(fused, x)).containsExactly(before);

			DeviceQ4KMatrix other = fused.rowSlice(128, 128);
			DeviceQ4KMatrix nested = other.rowSlice(0, 64);
			assertThat(mv.sgemv(nested, x)).containsExactly(Arrays.copyOfRange(before, 128, 192));
			fused.close();
			assertThat(other.isClosed()).isTrue();
			assertThat(nested.isClosed()).isTrue();
			assertThatThrownBy(other::devicePointer).isInstanceOf(IllegalStateException.class);
		} finally {
			fused.close();
		}
	}

	@Test
	@DisplayName("a range outside the matrix is refused")
	void rangesOutsideAreRefused() {
		try (DeviceQ4KMatrix fused = upload(256, QuantizationLayout.TYPE_Q4_K, new Random(10))) {
			assertThatThrownBy(() -> fused.rowSlice(200, 57)).isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> fused.rowSlice(-1, 10)).isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> fused.rowSlice(0, 0)).isInstanceOf(IllegalArgumentException.class);
		}
	}

	private static DeviceQ4KMatrix upload(int rows, int type, Random rnd) {
		byte[] raw = GgufKQuantCodec.encode(randomVec(rows * COLS, rnd), type);
		return DeviceQ4KMatrix.upload(ctx, raw, rows, COLS, type);
	}

	private static float[] randomVec(int n, Random rnd) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = rnd.nextFloat() * 2f - 1f;
		return a;
	}
}
