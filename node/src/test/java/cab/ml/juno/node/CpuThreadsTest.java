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

import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicIntegerArray;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The CPU thread count ({@code --threads}) and the pool that runs every CPU
 * matmul loop ({@link CpuThreads}, {@link CpuWorkerPool}).
 */
class CpuThreadsTest {

	private final int original = CpuThreads.threads();

	@AfterEach
	void restore() {
		CpuThreads.setThreads(original);
	}

	@Test
	void propertyBeatsEnvironmentBeatsDefault() {
		assertThat(CpuThreads.configured("3", "5", 12)).isEqualTo(3);
		assertThat(CpuThreads.configured(null, "5", 12)).isEqualTo(5);
		assertThat(CpuThreads.configured("  ", " 7 ", 12)).isEqualTo(7);
		assertThat(CpuThreads.configured(null, null, 12)).isEqualTo(12);
	}

	@Test
	void invalidCountsFailNamingTheirSource() {
		assertThatThrownBy(() -> CpuThreads.configured("0", null, 12)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("-Djuno.threads").hasMessageContaining("'0'");
		assertThatThrownBy(() -> CpuThreads.configured(null, "many", 12)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("JUNO_THREADS");
		assertThatThrownBy(() -> CpuThreads.setThreads(0)).isInstanceOf(IllegalArgumentException.class);
	}

	@ParameterizedTest(name = "threads={0}")
	@ValueSource(ints = { 1, 2, 12 })
	void everyIndexRunsExactlyOnce(int threads) {
		CpuThreads.setThreads(threads);
		for (int n : new int[] { 1, 7, 100, 10_000 }) {
			AtomicIntegerArray visits = new AtomicIntegerArray(n);
			CpuThreads.forEach(n, visits::incrementAndGet);
			for (int i = 0; i < n; i++)
				assertThat(visits.get(i)).as("n=%d index %d", n, i).isEqualTo(1);
		}
	}

	@ParameterizedTest(name = "threads={0}")
	@ValueSource(ints = { 2, 4 })
	void theThreadCountGovernsHowManyThreadsRunTheLoop(int threads) {
		CpuThreads.setThreads(threads);
		Set<Thread> seen = ConcurrentHashMap.newKeySet();
		CpuThreads.forEach(64, i -> {
			seen.add(Thread.currentThread());
			try {
				Thread.sleep(2);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		});
		assertThat(seen).as("distinct threads running a 64-index loop").hasSize(threads);
	}

	@Test
	void anExceptionOnAnyThreadReachesTheCaller() {
		CpuThreads.setThreads(4);
		assertThatThrownBy(() -> CpuThreads.forEach(1000, i -> {
			if (i == 777)
				throw new IllegalStateException("row 777");
		})).isInstanceOf(IllegalStateException.class).hasMessage("row 777");
		AtomicIntegerArray visits = new AtomicIntegerArray(10);
		CpuThreads.forEach(10, visits::incrementAndGet);
		assertThat(visits.get(9)).as("the pool still works after a failed loop").isEqualTo(1);
	}

	@Test
	void nestedAndConcurrentCallsAreCorrect() throws Exception {
		CpuThreads.setThreads(4);
		AtomicIntegerArray visits = new AtomicIntegerArray(40 * 40);
		CpuThreads.forEach(40, i -> CpuThreads.forEach(40, j -> visits.incrementAndGet(i * 40 + j)));
		for (int k = 0; k < visits.length(); k++)
			assertThat(visits.get(k)).isEqualTo(1);

		AtomicIntegerArray[] perCaller = new AtomicIntegerArray[3];
		Thread[] callers = new Thread[3];
		for (int c = 0; c < 3; c++) {
			AtomicIntegerArray v = perCaller[c] = new AtomicIntegerArray(5000);
			callers[c] = new Thread(() -> {
				for (int rep = 0; rep < 20; rep++)
					CpuThreads.forEach(5000, v::incrementAndGet);
			});
			callers[c].start();
		}
		for (Thread t : callers)
			t.join(60_000);
		for (AtomicIntegerArray v : perCaller)
			for (int i = 0; i < v.length(); i++)
				assertThat(v.get(i)).isEqualTo(20);
	}

	@Test
	void kernelResultsAreBitwiseIdenticalAtEveryThreadCount() {
		Random rnd = new Random(9);
		GgufReader.QuantizedTensor q4 = CpuIntegerKernelsTestSupport.synthetic(12, 300, 1024, rnd);
		GgufReader.QuantizedTensor q6 = CpuIntegerKernelsTestSupport.synthetic(14, 300, 1024, rnd);
		float[] x = new float[1024];
		float[][] X = new float[20][1024];
		for (int i = 0; i < 1024; i++)
			x[i] = (float) rnd.nextGaussian();
		for (float[] row : X)
			for (int i = 0; i < 1024; i++)
				row[i] = (float) rnd.nextGaussian();
		float[][] ref = null;
		for (int threads : new int[] { 1, 2, 12 }) {
			CpuThreads.setThreads(threads);
			float[] d4 = new float[300];
			float[] d6 = new float[300];
			CpuIntegerKernels.matVecInto(q4, x, 0, 300, d4, 1024);
			CpuIntegerKernels.matVecInto(q6, x, 0, 300, d6, 1024);
			float[][] g = new float[20][300];
			CpuIntegerGemm.sgemmInto(q4, X, g, 0, 300, 1024);
			float[][] got = { d4, d6, g[0], g[19] };
			if (ref == null)
				ref = got;
			else
				for (int k = 0; k < got.length; k++)
					assertThat(got[k]).as("threads=%d output %d", threads, k).isEqualTo(ref[k]);
		}
	}
}
