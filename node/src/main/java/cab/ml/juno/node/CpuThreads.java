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

import java.util.function.IntConsumer;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * The CPU thread count, and the one place that runs a CPU matmul loop in
 * parallel.
 *
 * <p>Every CPU matmul row loop (the integer and float-activation kernels, the
 * prefill matmul, the F32 sgemm the vision encoder uses, LoRA training's
 * transposes) goes through {@link #forEach}, which runs it on a
 * {@link CpuWorkerPool} of {@link #threads()} threads, the calling thread
 * included. The count comes from {@code --threads} (the launchers set
 * {@code -Djuno.threads}), else the {@code JUNO_THREADS} environment variable,
 * else the number of available processors; an embedder sets it with
 * {@link #setThreads}. It can change at any time: the pool is rebuilt at the
 * new size.
 *
 * <p>{@link Dispatch#COMMON_POOL} (parallel streams on the common pool, the
 * dispatch before the worker pool) remains selectable for measurement.
 */
public final class CpuThreads {

	private static final Logger log = Logger.getLogger(CpuThreads.class.getName());

	/** System property the launchers set from {@code --threads}. */
	public static final String PROPERTY = "juno.threads";

	/** Environment variable read when the property is absent. */
	public static final String ENV = "JUNO_THREADS";

	enum Dispatch {
		COMMON_POOL, WORKER_POOL
	}

	private static volatile Dispatch dispatch = Dispatch.WORKER_POOL;
	private static volatile CpuWorkerPool pool;
	private static volatile int threads = configured(System.getProperty(PROPERTY), System.getenv(ENV),
			Runtime.getRuntime().availableProcessors());

	private CpuThreads() {
	}

	/**
	 * The configured thread count: {@code property} if set, else {@code env}, else
	 * {@code fallback}.
	 *
	 * @throws IllegalArgumentException if the value that applies is not a positive integer
	 */
	static int configured(String property, String env, int fallback) {
		boolean fromProperty = property != null && !property.isBlank();
		String value = fromProperty ? property : env;
		if (value == null || value.isBlank())
			return fallback;
		value = value.trim();
		try {
			int n = Integer.parseInt(value);
			if (n >= 1)
				return n;
		} catch (NumberFormatException ignored) {
			// reported below
		}
		throw new IllegalArgumentException((fromProperty ? "-D" + PROPERTY : ENV) + " must be a positive integer, got '"
				+ value + "'");
	}

	/** Runs {@code body.accept(i)} for every {@code i} in {@code [0, n)} in parallel and returns when all have run. */
	static void forEach(int n, IntConsumer body) {
		if (dispatch == Dispatch.WORKER_POOL)
			pool().forEach(n, body);
		else
			IntStream.range(0, n).parallel().forEach(body);
	}

	/** Threads a CPU matmul runs on, the calling thread included. */
	public static int threads() {
		return threads;
	}

	/**
	 * Sets the CPU thread count for this JVM. Takes effect for the next matmul;
	 * the worker pool is rebuilt at the new size.
	 *
	 * @throws IllegalArgumentException if {@code n < 1}
	 */
	public static synchronized void setThreads(int n) {
		if (n < 1)
			throw new IllegalArgumentException("threads must be >= 1, got " + n);
		if (n == threads && pool != null)
			return;
		threads = n;
		CpuWorkerPool old = pool;
		pool = null;
		if (old != null)
			old.close();
	}

	private static CpuWorkerPool pool() {
		CpuWorkerPool p = pool;
		if (p == null) {
			synchronized (CpuThreads.class) {
				p = pool;
				if (p == null) {
					pool = p = new CpuWorkerPool(threads, "juno-cpu");
					log.info("CPU matmul threads: " + threads);
				}
			}
		}
		return p;
	}

	/** Selects the dispatch for the whole JVM; returns the previous one. For measurement and tests. */
	static Dispatch setDispatch(Dispatch d) {
		Dispatch was = dispatch;
		dispatch = d;
		return was;
	}
}
