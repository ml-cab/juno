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


import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * What {@code nvidia-smi} says about GPU use: the device memory held by the probe's own processes, the other
 * compute processes on the device, and utilization. Graphics programs do not appear as compute processes,
 * so utilization sampled before a GPU probe starts stands in for them.
 */
final class GpuLoad {

	/** Utilization above this, on the quietest of the samples taken before a GPU probe, means another program. */
	static final int BUSY_UTILIZATION = 20;
	static final List<String> COMPUTE_APPS = List.of("nvidia-smi",
			"--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits");
	static final List<String> UTILIZATION = List.of("nvidia-smi", "--query-gpu=utilization.gpu",
			"--format=csv,noheader,nounits");

	/** {@code others} as {@code "<pid> <process>"}. */
	record Sample(long ownMiB, List<String> others) {

		Sample {
			others = List.copyOf(others);
		}
	}

	private GpuLoad() {
	}

	/** {@code nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits}. */
	static Sample computeApps(String csv, Set<Long> ownPids) {
		long own = 0;
		List<String> others = new ArrayList<>();
		for (String line : csv.split("\n")) {
			String[] f = line.split(",");
			if (f.length < 3)
				continue;
			long pid;
			long mib;
			try {
				pid = Long.parseLong(f[0].trim());
				mib = Long.parseLong(f[f.length - 1].trim());
			} catch (NumberFormatException e) {
				continue;
			}
			if (ownPids.contains(pid))
				own += mib;
			else
				others.add(pid + " " + f[1].trim());
		}
		return new Sample(own, others);
	}

	/** The busiest device's utilization in percent, or -1 without a reading. */
	static int utilizationPercent(String csv) {
		int max = -1;
		for (String line : csv.split("\n")) {
			try {
				max = Math.max(max, Integer.parseInt(line.trim()));
			} catch (NumberFormatException ignored) {
			}
		}
		return max;
	}

	/** Busy when even the quietest sample is above {@link #BUSY_UTILIZATION}. */
	static boolean busy(List<Integer> samples) {
		int min = Integer.MAX_VALUE;
		for (int s : samples)
			if (s >= 0)
				min = Math.min(min, s);
		return min != Integer.MAX_VALUE && min > BUSY_UTILIZATION;
	}

	/** One {@code nvidia-smi} call, or an empty string where the tool is absent or fails. */
	static String query(List<String> cmd) {
		String out = KitBootstrap.runQuietly(Path.of(".").toAbsolutePath(), cmd);
		return out == null ? "" : out;
	}
}
