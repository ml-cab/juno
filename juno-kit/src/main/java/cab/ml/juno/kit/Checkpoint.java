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


import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Written after every probe, so an interrupted run resumes where it stopped and {@code pack} has
 * something to pack.
 *
 * @param selected probe ids in run order
 * @param done     outcomes so far, in the order they finished
 * @param args     the command line the run was started with
 */
record Checkpoint(int schemaVersion, String suite, List<String> selected, List<ProbeOutcome> done, List<String> args,
		String startedUtc) {

	static final String FILE = "checkpoint.json";

	Checkpoint {
		selected = List.copyOf(selected);
		done = List.copyOf(done);
		args = List.copyOf(args);
	}

	static Checkpoint start(String suite, List<String> selected, List<String> args, String startedUtc) {
		return new Checkpoint(1, suite, selected, List.of(), args, startedUtc);
	}

	Checkpoint with(ProbeOutcome outcome) {
		List<ProbeOutcome> d = new ArrayList<>(done);
		d.removeIf(o -> o.id().equals(outcome.id()));
		d.add(outcome);
		return new Checkpoint(schemaVersion, suite, selected, d, args, startedUtc);
	}

	List<String> remaining() {
		Set<String> finished = new HashSet<>();
		for (ProbeOutcome o : done)
			finished.add(o.id());
		return selected.stream().filter(id -> !finished.contains(id)).toList();
	}

	void save(Path runDir) throws IOException {
		Json.write(runDir.resolve(FILE), this);
	}

	static Checkpoint load(Path runDir) throws IOException {
		return Json.read(runDir.resolve(FILE), Checkpoint.class);
	}
}
