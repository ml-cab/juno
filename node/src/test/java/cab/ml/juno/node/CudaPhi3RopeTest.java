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

import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The Phi-3 family's extended rotation on the device against {@link Phi3Rope#ropeExt}
 * on the host, bit for bit: Phi-3.5-mini's shape of rope parameters (96-wide heads,
 * 48 frequency factors, a magnitude scale of about 1.19), a model without factors,
 * a multi-row window and a column range, and the positions the held-back long
 * factors would need refused as on the CPU.
 */
@Tag("gpu")
@DisplayName("CudaPhi3Rope - extended RoPE on a device-resident activation")
class CudaPhi3RopeTest {

	private static final int HEAD_DIM = 96;
	private static final int HEADS = 4;
	private static final int DIM = HEADS * HEAD_DIM;

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	/** Phi-3.5-mini's parameters: base 1e4, no linear scale, both factor sets, original context 4096. */
	private static Phi3RopeConfig phi35(Random rnd) {
		return new Phi3RopeConfig(10_000f, 1.0f, 1.1902381f, 4096, 131_072, randomVec(HEAD_DIM / 2, rnd, 1f, 1.6f),
				randomVec(HEAD_DIM / 2, rnd, 1f, 40f));
	}

	@Test
	@DisplayName("one row matches Phi3Rope.ropeExt bit for bit with frequency factors and a magnitude scale")
	void oneRowMatchesTheHost() {
		Random rnd = new Random(1);
		Phi3RopeConfig cfg = phi35(rnd);
		try (CudaPhi3Rope rope = CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, cfg); ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation x = chain.allocate(1, DIM);
			for (int pos : new int[] { 0, 1, 2, 17, 255, 511, 1000, 2047, 4095 }) {
				float[] host = randomVec(DIM, rnd, -3f, 3f);
				float[][] out = { null };
				x.upload(new float[][] { host.clone() });
				assertThat(rope.applyResident(x, pos)).isTrue();
				x.materialize(out);
				Phi3Rope.ropeExt(host, pos, HEADS, HEAD_DIM, cfg);
				assertThat(out[0]).as("pos %d", pos).containsExactly(host);
			}
		}
	}

	@Test
	@DisplayName("a model without factors, with a linear scale and no magnitude scale, matches too")
	void withoutFactorsMatchesTheHost() {
		Random rnd = new Random(2);
		Phi3RopeConfig cfg = new Phi3RopeConfig(500_000f, 0.25f, 1.0f, 4096, 4096, null, null);
		try (CudaPhi3Rope rope = CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, cfg); ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation x = chain.allocate(1, DIM);
			for (int pos : new int[] { 0, 3, 777, 4095, 6000 }) {
				float[] host = randomVec(DIM, rnd, -3f, 3f);
				float[][] out = { null };
				x.upload(new float[][] { host.clone() });
				assertThat(rope.applyResident(x, pos)).isTrue();
				x.materialize(out);
				Phi3Rope.ropeExt(host, pos, HEADS, HEAD_DIM, cfg);
				assertThat(out[0]).as("pos %d", pos).containsExactly(host);
			}
		}
	}

	@Test
	@DisplayName("a window of rows is rotated row r at startPos + r")
	void aWindowMatchesTheHostRowByRow() {
		Random rnd = new Random(3);
		Phi3RopeConfig cfg = phi35(rnd);
		int rows = 37;
		int startPos = 900;
		try (CudaPhi3Rope rope = CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, cfg); ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation x = chain.allocate(rows, DIM);
			float[][] host = new float[rows][];
			float[][] up = new float[rows][];
			for (int r = 0; r < rows; r++) {
				host[r] = randomVec(DIM, rnd, -3f, 3f);
				up[r] = host[r].clone();
			}
			x.upload(up);
			assertThat(rope.applyResident(x, startPos)).isTrue();
			float[][] out = new float[rows][];
			x.materialize(out);
			for (int r = 0; r < rows; r++) {
				Phi3Rope.ropeExt(host[r], startPos + r, HEADS, HEAD_DIM, cfg);
				assertThat(out[r]).as("row %d", r).containsExactly(host[r]);
			}
		}
	}

	@Test
	@DisplayName("a column range is rotated as whole heads and the columns around it are left alone")
	void aColumnRangeMatchesTheHost() {
		Random rnd = new Random(4);
		Phi3RopeConfig cfg = phi35(rnd);
		int width = 2 * HEAD_DIM;
		int from = HEAD_DIM;
		try (CudaPhi3Rope rope = CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, cfg); ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation x = chain.allocate(1, DIM);
			float[] host = randomVec(DIM, rnd, -3f, 3f);
			x.upload(new float[][] { host.clone() });
			assertThat(rope.applyResidentColumns(x, from, width, 321)).isTrue();
			float[][] out = { null };
			x.materialize(out);
			float[] cols = java.util.Arrays.copyOfRange(host, from, from + width);
			Phi3Rope.ropeExt(cols, 321, width / HEAD_DIM, HEAD_DIM, cfg);
			System.arraycopy(cols, 0, host, from, width);
			assertThat(out[0]).containsExactly(host);
		}
	}

	@Test
	@DisplayName("positions that need the held-back long factors are refused, as on the CPU")
	void refusesPositionsNeedingTheLongFactors() {
		Phi3RopeConfig cfg = phi35(new Random(5));
		assertThatThrownBy(() -> Phi3Rope.ropeExt(new float[DIM], 4096, HEADS, HEAD_DIM, cfg))
				.isInstanceOf(IllegalStateException.class);
		try (CudaPhi3Rope rope = CudaPhi3Rope.tryCreate(ctx, HEAD_DIM, cfg); ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation x = chain.allocate(2, DIM);
			x.upload(new float[][] { new float[DIM] });
			assertThatThrownBy(() -> rope.applyResident(x, 4096)).isInstanceOf(IllegalStateException.class)
					.hasMessageContaining("4096");
			assertThatThrownBy(() -> rope.applyResidentColumns(x, 0, HEAD_DIM, 5000))
					.isInstanceOf(IllegalStateException.class);
			x.upload(new float[][] { new float[DIM], new float[DIM] });
			assertThatThrownBy(() -> rope.applyResident(x, 4095)).as("the window's last row reaches 4096")
					.isInstanceOf(IllegalStateException.class);
		}
	}

	private static float[] randomVec(int n, Random rnd, float lo, float hi) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = lo + rnd.nextFloat() * (hi - lo);
		return a;
	}
}
