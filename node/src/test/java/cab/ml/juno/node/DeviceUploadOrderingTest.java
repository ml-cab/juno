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

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.MemorySegment;
import java.util.Random;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * A matrix is readable as soon as its upload returns. A host-to-device copy from pageable memory may return
 * before its transfer lands, and the GEMV kernels run on a stream that is not ordered after the copy's, so
 * without a wait the first GEMV can read part of the previous allocation's bytes. A laptop GPU (MX150, a
 * narrow link) failed {@link KQuantGemvRowsTest} that way at 512x2048, for Q5_K and Q6_K.
 *
 * <p>Two matrices of different data take turns in the same device allocation, so a GEMV that reads before
 * the upload lands reads the other matrix. Each upload is followed at once by a GEMV, compared bit for bit
 * with a GEMV after the device has finished all its work. The scratch is grown first, so no allocation in
 * between synchronizes the device by accident.
 */
@Tag("gpu")
@DisplayName("A device matrix reads correctly from the first GEMV after its upload")
class DeviceUploadOrderingTest {

	private static final int ROUNDS = 40;

	private static GpuContext ctx;
	private static CudaMatVec mv;

	@BeforeAll
	static void init() {
		assumeTrue(CudaAvailability.isAvailable(), "Skipping - no CUDA device");
		assumeTrue(CudaDriverBindings.isAvailable(), "No CUDA driver API - skipping");
		ctx = GpuContext.init(0);
		assumeTrue(Q4KMmqKernel.tryLoad() != null, "Q4_K MMQ kernel failed to load");
		mv = new CudaMatVec(ctx);
	}

	@AfterAll
	static void destroy() {
		if (mv != null)
			mv.releaseScratch();
		if (ctx != null)
			ctx.close();
	}

	@ParameterizedTest(name = "GGUF type {0}, {1}x{2}")
	@CsvSource({ "13, 512, 2048", "14, 512, 2048", "14, 4096, 4096", "12, 4096, 4096" })
	@DisplayName("the first GEMV after each upload equals a GEMV after the device is idle")
	void theFirstGemvAfterAnUploadReadsTheWholeMatrix(int type, int rows, int cols) {
		Random rnd = new Random(1234L + type + rows);
		byte[][] raw = { encoded(type, rows, cols, rnd), encoded(type, rows, cols, rnd) };
		float[] x = new float[cols];
		for (int i = 0; i < cols; i++)
			x[i] = -2f + 4f * rnd.nextFloat();
		try (DeviceQ4KMatrix warm = DeviceQ4KMatrix.upload(ctx, raw[0], rows, cols, type)) {
			mv.sgemv(warm, x); // grows the scratch to this shape
		}
		for (int round = 0; round < ROUNDS; round++) {
			try (DeviceQ4KMatrix a = DeviceQ4KMatrix.upload(ctx, raw[round & 1], rows, cols, type)) {
				float[] first = mv.sgemv(a, x);
				deviceIdle();
				float[] settled = mv.sgemv(a, x);
				for (int i = 0; i < rows; i++)
					if (Float.floatToRawIntBits(first[i]) != Float.floatToRawIntBits(settled[i]))
						throw new AssertionError("type " + type + " " + rows + "x" + cols + " round " + round
								+ ": row " + i + " read " + first[i] + " right after the upload, " + settled[i]
								+ " once the device was idle");
			}
		}
	}

	private static void deviceIdle() {
		GpuBindings gpu = ctx.bindings();
		GpuBindings.check(GpuBindings.callInt(gpu.gpuStreamSynchronize(), MemorySegment.NULL),
				"streamSynchronize(legacy default stream)");
	}

	private static byte[] encoded(int type, int rows, int cols, Random rnd) {
		float[] host = new float[rows * cols];
		for (int i = 0; i < host.length; i++)
			host[i] = -1f + 2f * rnd.nextFloat();
		return GgufKQuantCodec.encode(host, type);
	}
}
