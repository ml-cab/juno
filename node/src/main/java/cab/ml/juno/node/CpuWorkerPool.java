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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/**
 * A persistent pool of CPU worker threads for the matmul loops: no task objects,
 * no work stealing, a fixed thread count.
 *
 * <p>{@link #forEach} runs {@code [0, n)} on the calling thread and every
 * worker; each thread claims chunks of about {@code n / (8 * threads)} indices
 * from a shared counter until none are left, which balances load across
 * threads that share a core. A worker that finds no new job spins for
 * {@link #SPIN_NANOS}, so back-to-back matmuls (a decode step issues one per
 * projection) never pay a park and unpark, then parks until the next job.
 *
 * <p>One job runs at a time. A call from inside a running job (a worker, or the
 * caller that owns the job) runs inline on that thread. A call from another
 * thread waits for the running job to finish: the job already uses every
 * thread, so running two at once would only oversubscribe the cores. An
 * exception thrown by the body on any thread is rethrown to the caller once
 * every chunk has finished.
 */
final class CpuWorkerPool implements AutoCloseable {

	/** How long an idle worker spins before parking. */
	static final long SPIN_NANOS = 50_000L;

	private final int threads;
	private final Worker[] workers;
	private final ReentrantLock owner = new ReentrantLock();
	private final AtomicInteger next = new AtomicInteger();
	private final AtomicInteger remaining = new AtomicInteger();

	private volatile IntConsumer body;
	private volatile int n;
	private volatile int chunk;
	private volatile int generation;
	private volatile Throwable failure;
	private volatile boolean closed;

	/** A pool of {@code threads} threads in all: the caller plus {@code threads - 1} workers. */
	CpuWorkerPool(int threads, String name) {
		if (threads < 1)
			throw new IllegalArgumentException("threads must be >= 1, got " + threads);
		this.threads = threads;
		this.workers = new Worker[threads - 1];
		for (int i = 0; i < workers.length; i++) {
			workers[i] = new Worker(name + "-" + (i + 1));
			workers[i].start();
		}
	}

	int threads() {
		return threads;
	}

	/** Runs {@code body.accept(i)} for every {@code i} in {@code [0, n)} and returns when all have run. */
	void forEach(int n, IntConsumer body) {
		if (n <= 0)
			return;
		if (threads == 1 || n == 1 || Thread.currentThread() instanceof Worker || owner.isHeldByCurrentThread()) {
			for (int i = 0; i < n; i++)
				body.accept(i);
			return;
		}
		owner.lock();
		try {
			if (closed)
				throw new IllegalStateException("CPU worker pool is closed");
			this.failure = null;
			this.body = body;
			this.n = n;
			this.chunk = Math.max(1, n / (8 * threads));
			next.set(0);
			remaining.set(workers.length);
			generation++; // publishes the job to the spinning workers
			for (Worker w : workers)
				if (w.parked)
					LockSupport.unpark(w);
			try {
				claim(n, body);
			} catch (Throwable t) {
				failure = t;
			}
			while (remaining.get() > 0)
				Thread.onSpinWait();
			this.body = null;
			Throwable f = failure;
			if (f instanceof RuntimeException re)
				throw re;
			if (f instanceof Error e)
				throw e;
			if (f != null)
				throw new IllegalStateException(f);
		} finally {
			owner.unlock();
		}
	}

	private void claim(int n, IntConsumer body) {
		int c = chunk;
		for (int start = next.getAndAdd(c); start < n; start = next.getAndAdd(c)) {
			int end = Math.min(n, start + c);
			for (int i = start; i < end; i++)
				body.accept(i);
		}
	}

	/** Stops the workers after any running job. */
	@Override
	public void close() {
		owner.lock();
		try {
			closed = true;
			generation++;
			for (Worker w : workers)
				LockSupport.unpark(w);
		} finally {
			owner.unlock();
		}
	}

	private final class Worker extends Thread {
		volatile boolean parked;

		Worker(String name) {
			super(name);
			setDaemon(true);
		}

		@Override
		public void run() {
			int seen = 0;
			while (true) {
				int g = generation;
				if (g == seen) {
					long spinUntil = System.nanoTime() + SPIN_NANOS;
					while ((g = generation) == seen && System.nanoTime() < spinUntil)
						Thread.onSpinWait();
					if (g == seen) {
						parked = true;
						if (generation == seen)
							LockSupport.park(this);
						parked = false;
						continue;
					}
				}
				seen = g;
				if (closed)
					return;
				try {
					claim(n, body);
				} catch (Throwable t) {
					failure = t;
				}
				remaining.decrementAndGet();
			}
		}
	}
}
