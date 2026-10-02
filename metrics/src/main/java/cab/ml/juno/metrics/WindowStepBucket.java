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
package cab.ml.juno.metrics;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import jdk.jfr.consumer.RecordedEvent;

/**
 * Aggregates {@code juno.WindowStep}: the parts of a forward window that no per-op span
 * covers (the embedding, each projection call with its copy-out, bias adds, the KV write and
 * the LM head). Emits {@code juno.WindowStep.<step>.count}, {@code .prefill.count},
 * {@code .decode.count}, {@code .prefill.total_ms} and {@code .decode.total_ms}; every key of
 * the known steps is written on every run, zero or not, so a consumer never has to tell
 * "absent" from "none". A step outside the known list is reported under its own name.
 *
 * <p>A step belongs to prefill when its window is wider than one row, the rule
 * {@code juno.DeviceStaging} uses for the copies the same call issues.
 */
final class WindowStepBucket {

	static final String EVENT = "juno.WindowStep";

	static final List<String> KNOWN_STEPS = List.of("embed", "projection", "bias_add", "kv_write", "lm_head",
			"device_layer");

	private final Map<String, List<Long>> prefill = new TreeMap<>();
	private final Map<String, List<Long>> decode = new TreeMap<>();

	WindowStepBucket() {
		for (String s : KNOWN_STEPS) {
			prefill.put(s, new ArrayList<>());
			decode.put(s, new ArrayList<>());
		}
	}

	void accept(RecordedEvent ev, long nanos) {
		String step = ev.hasField("step") ? ev.getString("step") : null;
		if (step == null || step.isEmpty())
			step = "unknown";
		int windowSize = ev.hasField("windowSize") ? ev.getInt("windowSize") : 1;
		Map<String, List<Long>> phase = windowSize > 1 ? prefill : decode;
		phase.computeIfAbsent(step, k -> new ArrayList<>()).add(nanos);
		(phase == prefill ? decode : prefill).computeIfAbsent(step, k -> new ArrayList<>());
	}

	void putInto(Map<String, Double> m) {
		for (String step : prefill.keySet()) {
			List<Long> p = prefill.get(step);
			List<Long> d = decode.get(step);
			String prefix = EVENT + "." + step;
			m.put(prefix + ".count", (double) (p.size() + d.size()));
			m.put(prefix + ".prefill.count", (double) p.size());
			m.put(prefix + ".decode.count", (double) d.size());
			m.put(prefix + ".prefill.total_ms", JfrPercentiles.sumNanosToMs(p));
			m.put(prefix + ".decode.total_ms", JfrPercentiles.sumNanosToMs(d));
		}
	}
}
