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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The activation-residency primitive on its own: {@link ResidentChain} and
 * {@link ResidentActivation}, with the two operations that run on it today,
 * {@link CudaRmsNorm#normalizeResident} and {@link CudaRope#applyResident}.
 *
 * <p>Three properties are asserted here, because they are what the primitive is
 * for rather than incidental behaviour:
 * <ul>
 *   <li>a chain of two device operations, uploaded once and materialized once,
 *       computes what the scalar CPU path computes;</li>
 *   <li>the host sees device results only at {@link ResidentActivation#materialize},
 *       never by a device operation writing through to host memory;</li>
 *   <li>device memory returns exactly to its starting level after a chain is
 *       closed, asserted through {@link GpuBindings#memGetInfo} rather than
 *       inferred from not crashing - and that query is shown to see the
 *       allocation in the first place, so the no-leak assertion can fail.</li>
 * </ul>
 *
 * <p>{@code memGetInfo} is device-wide. Run on an otherwise idle device: another
 * process allocating during a run lands in the same figure.
 */
@Tag("gpu")
@DisplayName("ResidentChain / ResidentActivation - activation residency on the device")
class ResidentActivationTest {

	private static final int HEAD_DIM = 64;
	private static final float THETA = 10000f;
	private static final float EPS = 1e-5f;

	/** RMS norm differs from the scalar path by summation order only; RoPE matches to a float rounding. */
	private static final float CHAIN_TOL = 1e-4f;

	private static GpuContext ctx;
	private static CudaRmsNorm norm;
	private static CudaRope rope;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(RmsNormKernel.tryLoad() != null, "RMS-norm kernel failed to load");
		assumeTrue(RopeKernel.tryLoad() != null, "RoPE kernel failed to load");
		norm = CudaRmsNorm.tryCreate(ctx);
		rope = CudaRope.tryCreate(ctx, HEAD_DIM, THETA);
		assumeTrue(norm != null && rope != null, "CUDA backend required");
	}

	@AfterAll
	static void destroy() {
		if (rope != null)
			rope.close();
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("upload then materialize returns the uploaded rows bit for bit")
	void uploadThenMaterialize_isIdentity() {
		float[][] x = randomRows(5, 256, 1);
		try (ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation a = chain.allocate(8, 256);
			a.upload(x);
			assertThat(a.rows()).isEqualTo(5);
			float[][] out = new float[5][];
			a.materialize(out);
			for (int r = 0; r < 5; r++)
				assertThat(out[r]).as("row " + r).containsExactly(x[r]);
		}
	}

	@Test
	@DisplayName("norm then RoPE on the device, uploaded once and materialized once, matches the scalar chain")
	void normThenRope_onDevice_matchesScalarChain() {
		for (int rows : new int[] { 1, 7, 64 }) {
			int dim = 32 * HEAD_DIM;
			int startPos = 100;
			float[][] x = randomRows(rows, dim, 2 + rows);
			float[] weight = randomRows(1, dim, 99)[0];
			float[][] expected = scalarChain(x, weight, startPos);

			try (ResidentChain chain = ResidentChain.open(ctx);
					DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, weight, 1, dim)) {
				ResidentActivation in = chain.allocate(rows, dim);
				ResidentActivation normed = chain.allocate(rows, dim);
				in.upload(x);
				assertThat(norm.normalizeResident(in, w, EPS, normed)).isTrue();
				assertThat(rope.applyResident(normed, startPos)).isTrue();
				float[][] out = new float[rows][];
				normed.materialize(out);
				for (int r = 0; r < rows; r++)
					assertThat(out[r]).as("rows=" + rows + " row " + r).containsExactly(expected[r], within(CHAIN_TOL));
			}
		}
	}

	@Test
	@DisplayName("device operations never write through to host memory; results reach the host only at materialize")
	void deviceOperations_doNotTouchHostArrays() {
		int dim = 8 * HEAD_DIM;
		float[][] x = randomRows(3, dim, 3);
		float[][] xCopy = new float[3][];
		for (int r = 0; r < 3; r++)
			xCopy[r] = x[r].clone();
		float[] weight = randomRows(1, dim, 4)[0];

		try (ResidentChain chain = ResidentChain.open(ctx);
				DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, weight, 1, dim)) {
			ResidentActivation in = chain.allocate(3, dim);
			ResidentActivation normed = chain.allocate(3, dim);
			in.upload(x);
			norm.normalizeResident(in, w, EPS, normed);
			rope.applyResident(normed, 5);
			chain.sync();
			for (int r = 0; r < 3; r++)
				assertThat(x[r]).as("host input row " + r + " after device work").containsExactly(xCopy[r]);

			// The input activation still holds the upload: the norm wrote to a different buffer.
			float[][] stillX = new float[3][];
			in.materialize(stillX);
			for (int r = 0; r < 3; r++)
				assertThat(stillX[r]).containsExactly(xCopy[r]);
		}
	}

	@Test
	@DisplayName("a second upload before any synchronization cannot corrupt the first one's data in flight")
	void backToBackUploads_doNotRaceThePinnedStaging() {
		int rows = 512;
		int dim = 2048;
		float[][] first = randomRows(rows, dim, 5);
		float[][] second = randomRows(rows, dim, 6);
		float[] weight = randomRows(1, dim, 7)[0];
		float[][] expectedFirst = new float[rows][dim];
		for (int r = 0; r < rows; r++)
			LlamaTransformerHandler.rmsNormInto(first[r], weight, EPS, expectedFirst[r]);

		try (ResidentChain chain = ResidentChain.open(ctx);
				DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, weight, 1, dim)) {
			ResidentActivation busy = chain.allocate(rows, dim);
			ResidentActivation busyOut = chain.allocate(rows, dim);
			ResidentActivation in = chain.allocate(rows, dim);
			ResidentActivation normed = chain.allocate(rows, dim);
			busy.upload(second);
			chain.sync();
			// Queue milliseconds of device work ahead of the first upload, so its transfer
			// is still waiting in the stream when the host starts copying the second upload
			// into the same pinned buffer. Without the queue the transfer outruns the host
			// copy and the race cannot be observed; with it, an upload that does not wait
			// for its predecessor hands the device the second rows in place of the first.
			for (int i = 0; i < 500; i++)
				norm.normalizeResident(busy, w, EPS, busyOut);
			in.upload(first);
			norm.normalizeResident(in, w, EPS, normed);
			in.upload(second);
			float[][] out = new float[rows][];
			normed.materialize(out);
			for (int r = 0; r < rows; r++)
				assertThat(out[r]).as("row " + r + " normalized from the first upload")
						.containsExactly(expectedFirst[r], within(CHAIN_TOL));
		}
	}

	@Test
	@DisplayName("an open chain's allocation is visible to the device memory query, and closing returns it")
	void openChain_isVisibleToMemGetInfo_andCloseReturnsIt() {
		runChainCycle(); // first use of streams and kernels creates driver state that is kept
		long before = freeBytes();
		long during;
		long activationBytes;
		try (ResidentChain chain = ResidentChain.open(ctx)) {
			ResidentActivation a = chain.allocate(512, 2048);
			activationBytes = a.deviceBytes();
			assertThat(chain.deviceBytes()).isEqualTo(activationBytes);
			during = freeBytes();
		}
		long after = freeBytes();
		assertThat(activationBytes).isEqualTo(512L * 2048 * Float.BYTES);
		assertThat(before - during).as("free bytes consumed while the chain is open").isGreaterThanOrEqualTo(activationBytes);
		assertThat(after).as("free bytes after close").isEqualTo(before);
	}

	@Test
	@DisplayName("repeated allocate, chain and close cycles return device memory exactly")
	void repeatedCycles_leakNoDeviceMemory() {
		runChainCycle();
		long before = freeBytes();
		for (int cycle = 0; cycle < 3; cycle++)
			runChainCycle();
		assertThat(freeBytes()).as("free device memory after three chain cycles").isEqualTo(before);
	}

	@Test
	@DisplayName("closing the chain closes every activation on it, and close is idempotent")
	void closingChain_closesActivations() {
		ResidentChain chain = ResidentChain.open(ctx);
		ResidentActivation a = chain.allocate(2, 64);
		ResidentActivation b = chain.allocate(2, 64);
		b.close();
		b.close();
		chain.close();
		chain.close();
		assertThat(a.isClosed()).isTrue();
		assertThat(b.isClosed()).isTrue();
		assertThat(chain.isClosed()).isTrue();
		assertThat(chain.deviceBytes()).isZero();
		assertThatThrownBy(() -> a.upload(randomRows(1, 64, 8))).isInstanceOf(IllegalStateException.class);
		assertThatThrownBy(() -> chain.allocate(1, 64)).isInstanceOf(IllegalStateException.class);
	}

	@Test
	@DisplayName("uploads and operations outside an activation's shape are rejected, not truncated")
	void shapeViolations_areRejected() {
		int dim = 4 * HEAD_DIM;
		try (ResidentChain chain = ResidentChain.open(ctx);
				ResidentChain other = ResidentChain.open(ctx);
				DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, new float[dim], 1, dim);
				DeviceFloatMatrix shortW = DeviceFloatMatrix.upload(ctx, new float[dim / 2], 1, dim / 2)) {
			ResidentActivation a = chain.allocate(2, dim);
			ResidentActivation b = chain.allocate(2, dim);
			ResidentActivation narrow = chain.allocate(2, dim / 2);
			ResidentActivation elsewhere = other.allocate(2, dim);

			assertThatThrownBy(() -> a.upload(randomRows(3, dim, 9)))
					.as("more rows than capacity").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> a.upload(randomRows(1, dim + 1, 9)))
					.as("row wider than dim").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> a.materialize(new float[2][]))
					.as("nothing uploaded or written yet").isInstanceOf(IllegalStateException.class);
			assertThatThrownBy(() -> norm.normalizeResident(a, w, EPS, b))
					.as("norm of an activation holding no rows").isInstanceOf(IllegalStateException.class);

			a.upload(randomRows(2, dim, 10));
			assertThatThrownBy(() -> norm.normalizeResident(a, w, EPS, a))
					.as("in place would overwrite the residual stream").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> norm.normalizeResident(a, shortW, EPS, b))
					.as("weight length differs from dim").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> norm.normalizeResident(a, w, EPS, narrow))
					.as("output narrower than input").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> norm.normalizeResident(a, w, EPS, elsewhere))
					.as("output on another chain's stream").isInstanceOf(IllegalArgumentException.class);
			assertThatThrownBy(() -> rope.applyResident(chain.allocate(1, HEAD_DIM + 2), 0))
					.as("width not a multiple of the head size").isInstanceOf(IllegalArgumentException.class);
		}
	}

	/** Allocate, chain norm and RoPE on prefill-shaped rows, materialize, close. */
	private static void runChainCycle() {
		int rows = 64;
		int dim = 32 * HEAD_DIM;
		float[][] x = randomRows(rows, dim, 21);
		float[] weight = randomRows(1, dim, 22)[0];
		try (ResidentChain chain = ResidentChain.open(ctx);
				DeviceFloatMatrix w = DeviceFloatMatrix.upload(ctx, weight, 1, dim)) {
			ResidentActivation in = chain.allocate(rows, dim);
			ResidentActivation normed = chain.allocate(rows, dim);
			in.upload(x);
			norm.normalizeResident(in, w, EPS, normed);
			rope.applyResident(normed, 0);
			normed.materialize(new float[rows][]);
		}
	}

	private static long freeBytes() {
		long[] info = ctx.bindings().memGetInfo(ctx.deviceIndex());
		assumeTrue(info[0] > 0, "device memory query unavailable");
		return info[0];
	}

	/** The scalar path the handler runs today: rmsNormInto, then rope in place, per row. */
	private static float[][] scalarChain(float[][] x, float[] weight, int startPos) {
		int dim = weight.length;
		float[][] out = new float[x.length][dim];
		for (int r = 0; r < x.length; r++) {
			LlamaTransformerHandler.rmsNormInto(x[r], weight, EPS, out[r]);
			LlamaTransformerHandler.rope(out[r], startPos + r, dim / HEAD_DIM, HEAD_DIM, THETA);
		}
		return out;
	}

	private static float[][] randomRows(int rows, int dim, long seed) {
		Random rng = new Random(seed);
		float[][] x = new float[rows][dim];
		for (float[] row : x)
			for (int i = 0; i < dim; i++)
				row[i] = rng.nextFloat() * 4f - 2f;
		return x;
	}
}
