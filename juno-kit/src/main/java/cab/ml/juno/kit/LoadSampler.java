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

/**
 * Samples, once a second while a probe runs, how much of the machine's CPU other processes use: system CPU
 * time minus the probe's process tree and the kit's own. A process that exits keeps its last-seen CPU time,
 * so a short-lived child is not counted as another program. Where {@code nvidia-smi} runs it also samples,
 * every other second, the device memory the probe's processes hold and the other compute processes.
 */
final class LoadSampler implements AutoCloseable {

	private final com.sun.management.OperatingSystemMXBean os = (com.sun.management.OperatingSystemMXBean) ManagementFactory
			.getOperatingSystemMXBean();
	private final int cpus = Runtime.getRuntime().availableProcessors();
	private final ProcessHandle probe;
	private final Map<Long, Double> ownCpu = new HashMap<>();
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
		double ownBefore = ownCpuSeconds();
		long t0 = System.nanoTime();
		int tick = 0;
		while (!stop) {
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				return;
			}
			double load = os.getCpuLoad();
			double ownNow = ownCpuSeconds();
			long t1 = System.nanoTime();
			double dt = (t1 - t0) / 1e9;
			if (load >= 0)
				add(otherShareSeconds(load, cpus, dt, ownNow - ownBefore), dt);
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

	/** CPU seconds others used in an interval: system load times capacity, minus our own, never negative. */
	static double otherShareSeconds(double systemLoad, int cpus, double intervalSeconds, double ownCpuSeconds) {
		return Math.max(0, systemLoad * cpus * intervalSeconds - Math.max(0, ownCpuSeconds));
	}

	private double ownCpuSeconds() {
		record(ProcessHandle.current());
		record(probe);
		probe.descendants().forEach(this::record);
		double sum = 0;
		for (double v : ownCpu.values())
			sum += v;
		return sum;
	}

	private void record(ProcessHandle h) {
		h.info().totalCpuDuration().map(Duration::toNanos)
				.ifPresent(ns -> ownCpu.merge(h.pid(), ns / 1e9, Math::max));
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
