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
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

/**
 * The CPU integer and attention kernels keep every vector inside the method that computes it.
 *
 * <p>C2 boxes a vector passed to, or returned from, a method it does not inline, and the boxed value then takes the
 * Vector API's generic path, which allocates on every operation. Whether a helper is inlined depends on the size of its
 * caller and on the order in which methods were compiled, so a kernel that allocates nothing in a small test can
 * allocate tens of gigabytes per request in a server: the prefill matmul's one-row remainder loop did, through a
 * {@code widenPairs(ShortVector)} helper, once its four-row loop made the tile method large. The first check forbids
 * the shape outright: no production kernel class declares a method with a Vector API parameter or return type. The
 * second measures the matmul with remainder rows directly.
 */
class CpuKernelVectorEscapeTest {

	/** The production CPU kernels: the decode and prefill integer kernels at both widths and the attention head. */
	private static final Class<?>[] KERNELS = { CpuIntegerKernels.class, CpuIntegerGemm.class,
			CpuIntegerKernels256.class, CpuIntegerGemm256.class, GqaVectorHead.class };

	@Test
	void noKernelMethodTakesOrReturnsAVector() {
		List<String> escapes = new ArrayList<>();
		for (Class<?> k : KERNELS)
			for (Method m : k.getDeclaredMethods()) {
				if (isVector(m.getReturnType()))
					escapes.add(k.getSimpleName() + "." + m.getName() + " returns " + m.getReturnType().getSimpleName());
				for (Class<?> p : m.getParameterTypes())
					if (isVector(p))
						escapes.add(k.getSimpleName() + "." + m.getName() + " takes " + p.getSimpleName());
			}
		assertThat(escapes).as("methods through which a vector would cross a call").isEmpty();
	}

	@ParameterizedTest(name = "type {0}, B={1}")
	@CsvSource({ "12, 1", "12, 5", "12, 7", "14, 1", "14, 5", "14, 7", "13, 3", "8, 6" })
	void matmulWithRemainderRowsAllocatesNothingPerRow(int type, int batch) {
		int rows = 256, cols = 1024;
		Random rnd = new Random(7L * type + batch);
		GgufReader.QuantizedTensor w = CpuIntegerKernelsTest.synthetic(type, rows, cols, rnd);
		float[][] x = new float[batch][cols];
		for (float[] r : x)
			for (int i = 0; i < cols; i++)
				r[i] = (float) rnd.nextGaussian();
		float[][] y = new float[batch][rows];
		for (int i = 0; i < 400; i++) // compile the tile before measuring
			CpuIntegerGemm.sgemmInto(w, x, y, 0, rows, cols);
		com.sun.management.ThreadMXBean mx = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		int calls = 50;
		long before = allThreads(mx);
		for (int i = 0; i < calls; i++)
			CpuIntegerGemm.sgemmInto(w, x, y, 0, rows, cols);
		long perCall = (allThreads(mx) - before) / calls;
		// One boxed vector per sub-block of one row is already 256 rows x 4 blocks x 8 sub-blocks x 16 bytes = 128 KB
		// per call; task bookkeeping is a few hundred bytes per task.
		assertThat(perCall).as("bytes allocated per call, all threads").isLessThan(16 * 1024);
	}

	private static boolean isVector(Class<?> c) {
		return c.getName().startsWith("jdk.incubator.vector.");
	}

	private static long allThreads(com.sun.management.ThreadMXBean mx) {
		long total = 0;
		for (long b : mx.getThreadAllocatedBytes(mx.getAllThreadIds()))
			if (b > 0)
				total += b;
		return total;
	}
}
