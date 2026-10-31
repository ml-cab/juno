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


import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One entry of the probe catalog.
 *
 * @param suites         the suites that include it ({@code all} includes every probe)
 * @param models         model catalog ids it needs
 * @param refCostSeconds its duration on the reference host, which the calibration probe scales
 * @param priority       0 for probes every run needs; otherwise lower runs first under {@code --budget}
 * @param mainClass      simple name of the probe class {@link ProbeMain} dispatches to
 * @param args           arguments passed to the probe
 * @param heap           the child JVM's {@code -Xmx}
 */
record Probe(String id, String title, ProbeKind kind, Set<String> suites, Requirements requirements,
		List<String> models, int refCostSeconds, int priority, String mainClass, Map<String, String> args,
		int timeoutSeconds, String heap) {

	Probe {
		suites = Set.copyOf(suites);
		models = List.copyOf(models);
		args = Map.copyOf(args);
	}
}
