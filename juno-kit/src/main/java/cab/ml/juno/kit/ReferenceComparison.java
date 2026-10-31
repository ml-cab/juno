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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The correctness half of {@code ingest}: a returned run against the reference run at the same commit, probe by
 * probe. Mean NLL may be at most 1.01x the reference's, the bound the CPU-kernel quality reading used; argmax
 * agreement may fall by at most 0.02; checks and greedy replies are compared for equality. Throughput is not
 * compared: a remote reading never takes a ratio against the reference host.
 */
final class ReferenceComparison {

	static final double NLL_BOUND = 1.01;
	static final double AGREEMENT_DROP = 0.02;

	static final String EQUAL = "equal";
	static final String WITHIN_BOUND = "within the bound";
	static final String ABOVE_BOUND = "above 1.01x the reference";
	static final String LOWER = "more than 0.02 below the reference";
	static final String DIFFERS = "differs";
	static final String FAILS_HERE = "fails here, passes in the reference";
	static final String PASSES_HERE = "passes here, fails in the reference";

	/** One compared item; {@code remote} and {@code reference} are the values as text. */
	record Row(String probe, String item, String remote, String reference, String verdict) {

		/** Whether the draft lists it; equal items are only counted. */
		boolean listed() {
			return !verdict.equals(EQUAL) && !verdict.equals(WITHIN_BOUND);
		}
	}

	private ReferenceComparison() {
	}

	static List<Row> compare(RunResults remote, RunResults reference) {
		List<Row> rows = new ArrayList<>();
		for (Map.Entry<String, ProbeReport> e : remote.reports().entrySet()) {
			ProbeReport ref = reference.reports().get(e.getKey());
			if (ref != null)
				compare(e.getKey(), e.getValue(), ref, rows);
		}
		return rows;
	}

	private static void compare(String probe, ProbeReport here, ProbeReport ref, List<Row> rows) {
		Map<String, Boolean> refChecks = new HashMap<>();
		for (ProbeReport.Check c : ref.checks())
			refChecks.put(c.name(), c.passed());
		for (ProbeReport.Check c : here.checks()) {
			Boolean was = refChecks.get(c.name());
			if (was != null)
				rows.add(new Row(probe, c.name(), verdict(c.passed()), verdict(was),
						c.passed() == was ? EQUAL : c.passed() ? PASSES_HERE : FAILS_HERE));
		}

		Map<String, ProbeReport.Reading> refReadings = new HashMap<>();
		for (ProbeReport.Reading r : ref.readings())
			refReadings.put(r.name(), r);
		for (ProbeReport.Reading r : here.readings()) {
			ProbeReport.Reading was = refReadings.get(r.name());
			if (was == null)
				continue;
			double now = r.median(), then = was.median();
			if (r.name().startsWith("mean NLL"))
				rows.add(new Row(probe, r.name(), IngestReport.number(now), IngestReport.number(then),
						now <= NLL_BOUND * then ? WITHIN_BOUND : ABOVE_BOUND));
			else if (r.name().startsWith("argmax agreement"))
				rows.add(new Row(probe, r.name(), IngestReport.number(now), IngestReport.number(then),
						now >= then - AGREEMENT_DROP ? WITHIN_BOUND : LOWER));
		}

		for (Map.Entry<String, Object> f : here.facts().entrySet()) {
			if (!isReply(f.getKey()) || !(f.getValue() instanceof String now))
				continue;
			if (ref.facts().get(f.getKey()) instanceof String then)
				rows.add(new Row(probe, f.getKey(), now, then, now.equals(then) ? EQUAL : DIFFERS));
		}
	}

	/** The facts in which probes record a greedy reply: long-prompt texts, hot-path replies, vision captions. */
	static boolean isReply(String key) {
		return key.endsWith(".text") || key.endsWith(".short") || key.endsWith(".long") || key.endsWith("caption")
				|| key.endsWith("sentence");
	}

	private static String verdict(boolean passed) {
		return passed ? "PASS" : "FAIL";
	}
}
