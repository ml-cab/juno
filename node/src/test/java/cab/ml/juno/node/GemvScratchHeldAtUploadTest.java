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

import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A decode matrix-vector product allocates no device memory once its weights are
 * uploaded: the upload holds the single-row scratch the product needs.
 *
 * <p>The scratch used to grow on the first product. When several processes share a
 * device and fill it down to one reserve between them, that first growth found no
 * memory and failed the request, after the weights had loaded. Held at upload, the
 * scratch is part of the process's own footprint before the upload stop rule reads
 * the free memory.
 */
@Tag("gpu")
@DisplayName("Decode GEMV scratch is held at upload")
class GemvScratchHeldAtUploadTest {

	private static final int COLS = 512;
	private static final int[] ROWS = { 1376, 512, 512 };

	private static GpuContext ctx;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		ctx = GpuContext.init(0);
	}

	@AfterAll
	static void destroy() {
		if (ctx != null)
			ctx.close();
	}

	@Test
	@DisplayName("packed K-quant: sgemv and sgemvSameX leave the scratch as the uploads left it")
	void packedProductsAllocateNothingAfterUpload() {
		CudaMatVec mv = new CudaMatVec(ctx);
		assumeTrue(mv.supportsQ4KMmq(), "Skipping - K-quant MMQ kernel unavailable");
		Random rng = new Random(11);
		DeviceQ4KMatrix[] w = new DeviceQ4KMatrix[ROWS.length];
		try {
			for (int i = 0; i < w.length; i++)
				w[i] = mv.uploadKQuant(GgufKQuantCodec.encode(random(rng, ROWS[i] * COLS), QuantizationLayout.TYPE_Q4_K),
						ROWS[i], COLS, QuantizationLayout.TYPE_Q4_K);
			long held = mv.scratchDeviceBytes();
			assertThat(held).as("scratch held by the uploads").isPositive();
			float[] x = random(rng, COLS);
			for (DeviceQ4KMatrix m : w)
				mv.sgemv(m, x);
			mv.sgemvSameX(w, x);
			assertThat(mv.scratchDeviceBytes()).as("scratch after decode products").isEqualTo(held);
		} finally {
			for (DeviceQ4KMatrix m : w)
				if (m != null)
					m.close();
			mv.releaseScratch();
		}
	}

	@Test
	@DisplayName("FP16 resident: sgemv and sgemvSameX leave the scratch as the uploads left it")
	void halfProductsAllocateNothingAfterUpload() {
		CudaMatVec mv = new CudaMatVec(ctx);
		Random rng = new Random(12);
		DeviceHalfMatrix[] w = new DeviceHalfMatrix[ROWS.length];
		try {
			for (int i = 0; i < w.length; i++)
				w[i] = mv.uploadHalf(random(rng, ROWS[i] * COLS), ROWS[i], COLS);
			long held = mv.scratchDeviceBytes();
			assertThat(held).as("scratch held by the uploads").isPositive();
			float[] x = random(rng, COLS);
			for (DeviceHalfMatrix m : w)
				mv.sgemv(m, x);
			mv.sgemvSameX(w, x);
			assertThat(mv.scratchDeviceBytes()).as("scratch after decode products").isEqualTo(held);
		} finally {
			for (DeviceHalfMatrix m : w)
				if (m != null)
					m.close();
			mv.releaseScratch();
		}
	}

	@Test
	@DisplayName("FP32 resident: sgemv and sgemvSameX leave the scratch as the uploads left it")
	void floatProductsAllocateNothingAfterUpload() {
		CudaMatVec mv = new CudaMatVec(ctx);
		Random rng = new Random(13);
		DeviceFloatMatrix[] w = new DeviceFloatMatrix[ROWS.length];
		try {
			for (int i = 0; i < w.length; i++)
				w[i] = mv.upload(random(rng, ROWS[i] * COLS), ROWS[i], COLS);
			long held = mv.scratchDeviceBytes();
			assertThat(held).as("scratch held by the uploads").isPositive();
			float[] x = random(rng, COLS);
			for (DeviceFloatMatrix m : w)
				mv.sgemv(m, x);
			mv.sgemvSameX(w, x);
			assertThat(mv.scratchDeviceBytes()).as("scratch after decode products").isEqualTo(held);
		} finally {
			for (DeviceFloatMatrix m : w)
				if (m != null)
					m.close();
			mv.releaseScratch();
		}
	}

	private static float[] random(Random rng, int n) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = rng.nextFloat() * 2f - 1f;
		return a;
	}
}
