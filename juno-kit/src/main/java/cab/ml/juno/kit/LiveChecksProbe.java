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
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import cab.ml.juno.master.ModelLiveChecks;

/**
 * Juno's live checks on one model: pipeline-parallel (1 to 6) and tensor-parallel (7, 8) clusters of
 * forked node JVMs, a 512-token prefill on both schedules (9) and a context shift past the context limit
 * (10), on the backend the {@code backend} argument names: {@code cpu} keeps every check and node on the CPU,
 * {@code cuda} puts them on the GPU and fails closed without one, {@code auto} leaves each check its own
 * choice (CUDA wherever present).
 */
final class LiveChecksProbe implements KitProbe {

	@Override
	public ProbeReport run(String id, Map<String, String> args) throws Exception {
		String model = args.get("model");
		Set<ModelLiveChecks.Suite> suites = EnumSet.noneOf(ModelLiveChecks.Suite.class);
		for (String s : args.getOrDefault("suites", "PIPELINE,TENSOR,PREFILL,CONTEXT_SHIFT").split(","))
			suites.add(ModelLiveChecks.Suite.valueOf(s.trim().toUpperCase(Locale.ROOT)));
		ModelLiveChecks.Backend backend = ModelLiveChecks.Backend
				.valueOf(args.getOrDefault("backend", "auto").trim().toUpperCase(Locale.ROOT));
		List<ModelLiveChecks.LiveCheck> results = ModelLiveChecks.run(model, suites, backend, System.out);
		List<ProbeReport.Check> checks = new ArrayList<>();
		boolean all = !results.isEmpty();
		for (ModelLiveChecks.LiveCheck c : results) {
			checks.add(new ProbeReport.Check(c.number() + ". " + c.name(), c.passed(), c.detail()));
			all &= c.passed();
		}
		return new ProbeReport(id, ProbeKind.CHECK, all, checks, List.of(),
				Map.of("architecture", ModelLiveChecks.architecture(model), "suites", suites.toString(), "backend",
						backend.toString()));
	}
}
