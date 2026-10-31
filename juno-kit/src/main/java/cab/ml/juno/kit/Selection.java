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


import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Turns a suite or a probe list, the detected host and an optional time budget into what runs and what does not. */
final class Selection {

	static final List<String> SUITES = List.of("quick", "cpu", "gpu", "all", "custom");

	/** A probe that will not run, and why. */
	record Skip(Probe probe, String reason) {
	}

	/** {@code selected} keeps catalog order, so the environment and calibration probes run first. */
	record Result(List<Probe> selected, List<Skip> skipped) {
	}

	private Selection() {
	}

	/**
	 * @param models  catalog model ids that are present or can be downloaded
	 * @param budget  {@code null} for no budget; otherwise probes are kept in priority order while their
	 *                reference cost times {@code scale} fits, and priority-0 probes are always kept
	 */
	static Result select(List<Probe> catalog, String suite, List<String> probeIds, HostInfo host, Set<String> models,
			Duration budget, double scale) {
		List<Probe> candidates = candidates(catalog, suite, probeIds);
		List<Probe> runnable = new ArrayList<>();
		List<Skip> skipped = new ArrayList<>();
		for (Probe p : candidates) {
			String reason = p.requirements().unmetBy(host);
			if (reason == null)
				reason = missingModel(p, models);
			if (reason == null)
				runnable.add(p);
			else
				skipped.add(new Skip(p, reason));
		}
		if (budget == null)
			return new Result(runnable, skipped);

		Set<String> kept = new HashSet<>();
		double left = budget.toSeconds();
		List<Probe> byPriority = new ArrayList<>(runnable);
		byPriority.sort(Comparator.comparingInt(Probe::priority).thenComparing(Probe::id));
		for (Probe p : byPriority) {
			double cost = p.refCostSeconds() * scale;
			if (p.priority() == 0 || cost <= left) {
				kept.add(p.id());
				left -= cost;
			}
		}
		List<Probe> selected = new ArrayList<>();
		for (Probe p : runnable) {
			if (kept.contains(p.id()))
				selected.add(p);
			else
				skipped.add(new Skip(p, "over the --budget of " + budget.toMinutes() + " min"));
		}
		return new Result(selected, skipped);
	}

	private static List<Probe> candidates(List<Probe> catalog, String suite, List<String> probeIds) {
		if (!SUITES.contains(suite))
			throw new IllegalArgumentException("unknown suite '" + suite + "'; choose one of " + SUITES);
		if (!suite.equals("custom")) {
			List<Probe> out = new ArrayList<>();
			for (Probe p : catalog)
				if (suite.equals("all") || p.suites().contains(suite))
					out.add(p);
			return out;
		}
		if (probeIds.isEmpty())
			throw new IllegalArgumentException("suite custom needs --probes <id,id,...>");
		Set<String> wanted = new LinkedHashSet<>(probeIds);
		List<String> known = catalog.stream().map(Probe::id).toList();
		for (String id : wanted)
			if (!known.contains(id))
				throw new IllegalArgumentException("unknown probe '" + id + "'; known probes: " + known);
		return catalog.stream().filter(p -> wanted.contains(p.id())).toList();
	}

	private static String missingModel(Probe p, Set<String> models) {
		for (String m : p.models())
			if (!models.contains(m))
				return "model " + m + " is neither present nor downloadable";
		return null;
	}
}
