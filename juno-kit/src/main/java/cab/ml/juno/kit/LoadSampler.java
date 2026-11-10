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
package cab.ml.juno.kit;


import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Samples, once a second while a probe runs, how much of the machine's CPU other processes use: system CPU
 * time minus the probe's process tree and the kit's own, each process's CPU time read at both ends of the
 * second. A second in which one of those processes exited is left out: the CPU it used after its last sample
 * is in the system's load and in no sample of it, so it would count as another program's. A probe that starts
 * many short child JVMs (the kernel width probe starts 24 per attempt) read several points high before this. Where {@code nvidia-smi} runs it also samples,
 * every other second, the device memory the probe's processes hold and the other compute processes.
 */
final class LoadSampler implements AutoCloseable {

	private final com.sun.management.OperatingSystemMXBean os = (com.sun.management.OperatingSystemMXBean) ManagementFactory
			.getOperatingSystemMXBean();
	private final int cpus = Runtime.getRuntime().availableProcessors();
	private final ProcessHandle probe;
	private final Thread thread;
	private final boolean gpu;
	private volatile boolean stop;
	private double otherCpuSeconds;
	private double wallSeconds;
	private long ownGpuPeakMiB;
	private final java.util.Set<String> otherGpuProcesses = new java.util.LinkedHashSet<>();

	LoadSampler(ProcessHandle probe, boolean gpu) {
		this.probe = probe;
		this.gpu = gpu;
		this.thread = Thread.ofPlatform().daemon().name("juno-kit-load").start(this::loop);
	}

	private void loop() {
		os.getCpuLoad();
		Map<Long, Double> ownBefore = ownCpu();
		long t0 = System.nanoTime();
		int tick = 0;
		while (!stop) {
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				return;
			}
			double load = os.getCpuLoad();
			Map<Long, Double> ownNow = ownCpu();
			long t1 = System.nanoTime();
			double dt = (t1 - t0) / 1e9;
			if (load >= 0)
				otherCpuSeconds(load, cpus, dt, ownBefore, ownNow).ifPresent(other -> add(other, dt));
			ownBefore = ownNow;
			t0 = t1;
			if (gpu && tick++ % 2 == 0)
				sampleGpu();
		}
	}

	private void sampleGpu() {
		java.util.Set<Long> pids = new java.util.HashSet<>();
		pids.add(ProcessHandle.current().pid());
		pids.add(probe.pid());
		probe.descendants().forEach(h -> pids.add(h.pid()));
		GpuLoad.Sample s = GpuLoad.computeApps(GpuLoad.query(GpuLoad.COMPUTE_APPS), pids);
		synchronized (this) {
			ownGpuPeakMiB = Math.max(ownGpuPeakMiB, s.ownMiB());
			otherGpuProcesses.addAll(s.others());
		}
	}

	synchronized long ownGpuPeakMiB() {
		return ownGpuPeakMiB;
	}

	/** Other compute processes seen on the GPU while the probe ran, as {@code "<pid> <process>"}. */
	synchronized java.util.List<String> otherGpuProcesses() {
		return java.util.List.copyOf(otherGpuProcesses);
	}

	private synchronized void add(double other, double dt) {
		otherCpuSeconds += other;
		wallSeconds += dt;
	}

	/**
	 * Other programs' CPU seconds in one interval, from the CPU seconds so far of the probe's live processes at its
	 * start ({@code before}) and end ({@code now}), by PID. Empty when a process in {@code before} is gone from
	 * {@code now}: it exited during the interval, and its CPU after the first sample was never read. A process new
	 * in {@code now} started during the interval and counts from zero.
	 */
	static OptionalDouble otherCpuSeconds(double systemLoad, int cpus, double intervalSeconds, Map<Long, Double> before,
			Map<Long, Double> now) {
		if (!now.keySet().containsAll(before.keySet()))
			return OptionalDouble.empty();
		double own = 0;
		for (Map.Entry<Long, Double> e : now.entrySet())
			own += e.getValue() - before.getOrDefault(e.getKey(), 0.0);
		return OptionalDouble.of(otherShareSeconds(systemLoad, cpus, intervalSeconds, own));
	}

	/** CPU seconds others used in an interval: system load times capacity, minus our own, never negative. */
	static double otherShareSeconds(double systemLoad, int cpus, double intervalSeconds, double ownCpuSeconds) {
		return Math.max(0, systemLoad * cpus * intervalSeconds - Math.max(0, ownCpuSeconds));
	}

	/** CPU seconds so far of this JVM, the probe and the probe's live descendants, by PID. */
	private Map<Long, Double> ownCpu() {
		Map<Long, Double> cpu = new HashMap<>();
		record(cpu, ProcessHandle.current());
		record(cpu, probe);
		probe.descendants().forEach(h -> record(cpu, h));
		return cpu;
	}

	private static void record(Map<Long, Double> cpu, ProcessHandle h) {
		h.info().totalCpuDuration().map(Duration::toNanos).ifPresent(ns -> cpu.put(h.pid(), ns / 1e9));
	}

	/** Others' share of the machine's CPU over the probe so far, 0 to 1. */
	synchronized double otherCpuShare() {
		return wallSeconds == 0 ? 0 : otherCpuSeconds / (wallSeconds * cpus);
	}

	@Override
	public void close() {
		stop = true;
		thread.interrupt();
	}
}
