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

import java.lang.management.ManagementFactory;
import java.util.Random;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The child JVM of {@link CpuGemmProfilePollutionTest}. It runs the Vector API operations the batched matmul uses at
 * 128 bits until they are hot, then at 64 and 256 bits a few thousand times, so the JDK's shared methods have profiled
 * a second vector shape. Then it runs prefill-shaped windows through {@link CpuIntegerGemm} (four TinyLlama-sized
 * layers of four Q4_K 2048x2048 matmuls and one Q6_K 2048x5632, at 572 activation rows) and prints the bytes allocated
 * per matmul call on all threads in the last window: {@code bytes/call <n>}.
 */
public final class CpuGemmProfilePollutionMain {

	private CpuGemmProfilePollutionMain() {
	}

	public static void main(String[] args) {
		float s = shapes(ShortVector.SPECIES_128, FloatVector.SPECIES_128, 200_000)
				+ shapes(ShortVector.SPECIES_64, FloatVector.SPECIES_64, 2_000)
				+ shapes(ShortVector.SPECIES_256, FloatVector.SPECIES_256, 2_000);
		System.out.println("other shapes: " + s);

		Random rnd = new Random(1);
		GgufReader.QuantizedTensor q4 = CpuIntegerKernelsTestSupport.synthetic(12, 2048, 2048, rnd);
		GgufReader.QuantizedTensor q6 = CpuIntegerKernelsTestSupport.synthetic(14, 2048, 5632, rnd);
		float[][] x4 = rows(572, 2048, rnd), x6 = rows(572, 5632, rnd);
		float[][] y = new float[572][2048];
		com.sun.management.ThreadMXBean mx = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
		long perCall = 0;
		for (int window = 0; window < 3; window++) {
			long before = allThreads(mx);
			for (int layer = 0; layer < 4; layer++) {
				for (int k = 0; k < 4; k++)
					CpuIntegerGemm.sgemmInto(q4, x4, y, 0, 2048, 2048);
				CpuIntegerGemm.sgemmInto(q6, x6, y, 0, 2048, 5632);
			}
			perCall = (allThreads(mx) - before) / 20;
		}
		System.out.println("bytes/call " + perCall);
		System.exit(0);
	}

	/** The batched matmul's lane operations at one shape: short multiply-add, int shifts, I2F, float multiply, add, sub. */
	private static float shapes(VectorSpecies<Short> ss, VectorSpecies<Float> fs, int n) {
		short[] w = new short[64];
		short[] x = new short[64];
		float[] o = new float[64];
		for (int i = 0; i < 64; i++) {
			w[i] = (short) (i - 31);
			x[i] = (short) (3 * i - 90);
			o[i] = i * 0.25f;
		}
		FloatVector acc = FloatVector.zero(fs);
		for (int i = 0; i < n; i++) {
			ShortVector p = ShortVector.fromArray(ss, w, 0).mul(ShortVector.fromArray(ss, x, 0))
					.add(ShortVector.fromArray(ss, w, ss.length()).mul(ShortVector.fromArray(ss, x, ss.length())));
			IntVector v = p.reinterpretAsInts();
			acc = acc.add(((FloatVector) v.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
					.add(v.lanewise(VectorOperators.ASHR, 16)).convertShape(VectorOperators.I2F, fs, 0)).mul(1e-6f));
			acc = acc.sub(FloatVector.fromArray(fs, o, 0).mul(FloatVector.fromArray(fs, o, fs.length())));
		}
		return acc.lane(0);
	}

	private static float[][] rows(int n, int cols, Random rnd) {
		float[][] x = new float[n][cols];
		for (float[] r : x)
			for (int i = 0; i < cols; i++)
				r[i] = (float) rnd.nextGaussian();
		return x;
	}

	private static long allThreads(com.sun.management.ThreadMXBean mx) {
		long total = 0;
		for (long b : mx.getThreadAllocatedBytes(mx.getAllThreadIds()))
			if (b > 0)
				total += b;
		return total;
	}
}
