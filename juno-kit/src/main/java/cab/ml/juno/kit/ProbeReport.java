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
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * What a probe's child JVM writes: pass or fail per check, and per reading every repetition's value.
 *
 * @param passed {@code null} for a probe with no checks
 * @param facts  anything else worth keeping (shapes, token counts, JVM settings)
 */
record ProbeReport(String id, ProbeKind kind, Boolean passed, List<Check> checks, List<Reading> readings,
		Map<String, Object> facts) {

	ProbeReport {
		checks = List.copyOf(checks);
		readings = List.copyOf(readings);
		facts = Map.copyOf(facts);
	}

	record Check(String name, boolean passed, String detail) {
	}

	/**
	 * One measured quantity; {@code values} holds every repetition, in the order taken. Only a
	 * {@code scored} reading is held to the noise rule: throughput is, while GC pause totals and
	 * allocation, whose near-zero values move by large ratios, are published for reading.
	 */
	record Reading(String name, String unit, List<Double> values, boolean higherIsBetter, boolean scored) {

		Reading {
			values = List.copyOf(values);
		}

		double median() {
			return sorted().isEmpty() ? Double.NaN : sorted().get(sorted().size() / 2);
		}

		double min() {
			return sorted().isEmpty() ? Double.NaN : sorted().get(0);
		}

		double max() {
			return sorted().isEmpty() ? Double.NaN : sorted().get(sorted().size() - 1);
		}

		private List<Double> sorted() {
			List<Double> s = new ArrayList<>(values);
			Collections.sort(s);
			return s;
		}
	}
}
