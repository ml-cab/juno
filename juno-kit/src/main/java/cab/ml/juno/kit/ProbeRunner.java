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


import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import cab.ml.juno.player.ClusterHarness;

/**
 * Runs each probe in a fresh JVM with Juno's runtime flags and a timeout, logging its output to
 * {@code probes/<id>.log}. A crash, a non-zero exit or a timeout is recorded and the run goes on. A reading
 * the noise rule flags is re-run up to twice; earlier attempts' reports are kept beside the final one.
 */
final class ProbeRunner implements RunSession.Executor {

	private final Map<String, Path> modelPaths;
	private final boolean nvidiaSmi;
	private volatile Process running;

	ProbeRunner(Map<String, Path> modelPaths) {
		this.modelPaths = Map.copyOf(modelPaths);
		this.nvidiaSmi = !GpuLoad.query(GpuLoad.UTILIZATION).isBlank();
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			Process p = running;
			if (p != null)
				kill(p);
		}));
	}

	@Override
	public ProbeOutcome execute(Probe probe, Path runDir) throws Exception {
		Path dir = runDir.resolve(RunSession.PROBES_DIR);
		Files.createDirectories(dir);
		Path report = dir.resolve(probe.id() + ".json");
		Path log = dir.resolve(probe.id() + ".log");
		long t0 = System.nanoTime();
		StringBuilder reruns = new StringBuilder();
		List<String> props = new ArrayList<>();
		String root = System.getProperty("juno.kit.root");
		if (root != null)
			props.add("-Djuno.kit.root=" + root);
		props.add(errorFileOption(dir, probe.id()));
		Path nodeLogs = nodeLogDir(dir, probe.id());
		for (int attempt = 1;; attempt++) {
			Files.deleteIfExists(report);
			boolean busyBefore = probe.requirements().needsGpu() && nvidiaSmi && gpuBusyBeforeStart();
			List<String> args = probeArgs(probe, report, modelPaths, dir);
			List<String> cmd = JvmLauncher.command(probe.heap(), probe.kind() == ProbeKind.READING, props,
					ProbeMain.class.getName(), args);
			Files.writeString(log, "== attempt " + attempt + ": " + ClusterHarness.NODE_LOG_DIR_ENV + "=" + nodeLogs
					+ " " + String.join(" ", cmd) + "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
			Process p = probeProcess(cmd, log, nodeLogs).start();
			p.getOutputStream().close();
			running = p;
			boolean finished;
			double share;
			long gpuMiB;
			List<String> otherGpu;
			try (LoadSampler load = new LoadSampler(p.toHandle(), nvidiaSmi)) {
				finished = p.waitFor(probe.timeoutSeconds(), TimeUnit.SECONDS);
				share = load.otherCpuShare();
				gpuMiB = load.ownGpuPeakMiB();
				otherGpu = load.otherGpuProcesses();
			} catch (InterruptedException e) {
				kill(p);
				throw e;
			} finally {
				running = null;
			}
			long wall = (System.nanoTime() - t0) / 1_000_000;
			boolean gpuBusy = busyBefore || (probe.requirements().needsGpu() && !otherGpu.isEmpty());
			String gpuNote = otherGpu.isEmpty() ? "" : "other GPU processes: " + String.join(", ", otherGpu) + ". ";
			String logRef = "see " + RunSession.PROBES_DIR + "/" + log.getFileName();
			if (!finished) {
				kill(p);
				return new ProbeOutcome(probe.id(), ProbeOutcome.TIMEOUT, wall, attempt,
						"no result after " + probe.timeoutSeconds() + " s; " + logRef, share, gpuMiB, gpuBusy);
			}
			if (p.exitValue() != 0 || !Files.isRegularFile(report))
				return new ProbeOutcome(probe.id(), ProbeOutcome.ERROR, wall, attempt,
						"exit code " + p.exitValue() + "; " + logRef, share, gpuMiB, gpuBusy);
			ProbeReport r = Json.read(report, ProbeReport.class);
			NoiseRule.Verdict v = NoiseRule.verdict(r, share, gpuBusy, attempt);
			if (v == NoiseRule.Verdict.RERUN) {
				Files.move(report, dir.resolve(probe.id() + ".attempt" + attempt + ".json"),
						StandardCopyOption.REPLACE_EXISTING);
				reruns.append("attempt ").append(attempt).append(" re-run: ")
						.append(NoiseRule.reason(r, share, gpuBusy)).append(". ");
				continue;
			}
			if (r.kind() == ProbeKind.CHECK) {
				boolean pass = Boolean.TRUE.equals(r.passed());
				return new ProbeOutcome(probe.id(), pass ? ProbeOutcome.PASS : ProbeOutcome.FAIL, wall, attempt,
						(gpuNote + (pass ? "" : failed(r))).trim(), share, gpuMiB, gpuBusy);
			}
			return new ProbeOutcome(probe.id(), v == NoiseRule.Verdict.NOISY ? ProbeOutcome.NOISY : ProbeOutcome.DONE,
					wall, attempt,
					(reruns + gpuNote + (v == NoiseRule.Verdict.NOISY ? "still: " + NoiseRule.reason(r, share, gpuBusy)
							: "")).trim(),
					share, gpuMiB, gpuBusy);
		}
	}

	/**
	 * The probe's JVM, its output appended to its log. Every Juno JVM it starts inherits
	 * {@link ClusterHarness#NODE_LOG_DIR_ENV}, so the cluster nodes those JVMs fork write their standard error and
	 * JVM error files into {@code nodeLogs}, inside the run folder, which is packed and scrubbed with the rest.
	 */
	static ProcessBuilder probeProcess(List<String> cmd, Path log, Path nodeLogs) {
		ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true)
				.redirectOutput(ProcessBuilder.Redirect.appendTo(log.toFile()));
		pb.environment().put(ClusterHarness.NODE_LOG_DIR_ENV, nodeLogs.toString());
		return pb;
	}

	/** The folder for the forked cluster nodes of probe {@code id}: {@code probes/<id>/nodes}. */
	static Path nodeLogDir(Path probesDir, String id) {
		return probesDir.resolve(id).resolve("nodes");
	}

	/**
	 * The JVM option that puts probe {@code id}'s own error file in the probes folder, which exists before the
	 * probe starts; HotSpot otherwise writes it to the working directory, the tester's checkout.
	 */
	static String errorFileOption(Path probesDir, String id) {
		return "-XX:ErrorFile=" + probesDir.resolve(id + ".hs_err_pid%p.log");
	}

	/**
	 * {@link ProbeMain}'s arguments: the probe's own, its first model as {@code model}, every model as
	 * {@code model.<catalog id>}, and the directory for its side files.
	 */
	static List<String> probeArgs(Probe probe, Path report, Map<String, Path> modelPaths, Path logDir) {
		List<String> args = new ArrayList<>(List.of(probe.mainClass(), probe.id(), report.toString()));
		probe.args().forEach((k, v) -> args.add(k + "=" + v));
		if (!probe.models().isEmpty())
			args.add("model=" + modelPaths.get(probe.models().get(0)));
		for (String m : probe.models())
			args.add("model." + m + "=" + modelPaths.get(m));
		args.add("logDir=" + logDir);
		return args;
	}

	/** Three utilization samples a second apart before a GPU probe starts; busy if even the quietest is high. */
	private static boolean gpuBusyBeforeStart() throws InterruptedException {
		List<Integer> samples = new ArrayList<>();
		for (int i = 0; i < 3; i++) {
			if (i > 0)
				Thread.sleep(1000);
			samples.add(GpuLoad.utilizationPercent(GpuLoad.query(GpuLoad.UTILIZATION)));
		}
		return GpuLoad.busy(samples);
	}

	private static String failed(ProbeReport r) {
		List<String> names = new ArrayList<>();
		for (ProbeReport.Check c : r.checks())
			if (!c.passed())
				names.add(c.name());
		return names.isEmpty() ? "no checks ran" : "failed: " + String.join("; ", names);
	}

	private static void kill(Process p) {
		p.descendants().forEach(ProcessHandle::destroyForcibly);
		p.destroyForcibly();
	}
}
