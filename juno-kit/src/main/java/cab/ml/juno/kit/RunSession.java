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
import java.util.List;

/** Runs the probes a checkpoint has not finished, in order, saving the checkpoint after each one. */
final class RunSession {

	static final String PROBES_DIR = "probes";

	/** Runs one probe and reports its outcome; an {@link InterruptedException} stops the run. */
	interface Executor {
		ProbeOutcome execute(Probe probe, Path runDir) throws Exception;
	}

	interface Listener {
		Listener NONE = new Listener() {
		};

		default void started(Probe probe, int index, int total) {
		}

		default void finished(Probe probe, ProbeOutcome outcome, int index, int total) {
		}
	}

	private final Path runDir;
	private final List<Probe> selected;
	private final Executor executor;
	private final Listener listener;
	private Checkpoint checkpoint;

	RunSession(Path runDir, List<Probe> selected, Checkpoint checkpoint, Executor executor, Listener listener) {
		this.runDir = runDir;
		this.selected = List.copyOf(selected);
		this.checkpoint = checkpoint;
		this.executor = executor;
		this.listener = listener;
	}

	Checkpoint run() throws Exception {
		Files.createDirectories(runDir.resolve(PROBES_DIR));
		checkpoint.save(runDir);
		List<String> remaining = checkpoint.remaining();
		int total = selected.size();
		int index = total - remaining.size();
		for (Probe probe : selected) {
			if (!remaining.contains(probe.id()))
				continue;
			index++;
			listener.started(probe, index, total);
			long t0 = System.nanoTime();
			ProbeOutcome outcome;
			try {
				outcome = executor.execute(probe, runDir);
			} catch (InterruptedException e) {
				throw e;
			} catch (Exception e) {
				outcome = new ProbeOutcome(probe.id(), ProbeOutcome.ERROR, (System.nanoTime() - t0) / 1_000_000, 1,
						e.toString(), 0, 0, false);
			}
			checkpoint = checkpoint.with(outcome);
			checkpoint.save(runDir);
			listener.finished(probe, outcome, index, total);
		}
		return checkpoint;
	}
}
