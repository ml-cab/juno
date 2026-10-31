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
import java.util.List;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/** Counts one event type across JFR recordings, as the smoke scripts read it from the metrics summary. */
final class JfrEvents {

	static final String GRAMMAR_CONSTRAINED = "juno.GrammarConstrained";

	private JfrEvents() {
	}

	static long count(List<Path> recordings, String eventName) throws IOException {
		long n = 0;
		for (Path f : recordings)
			try (RecordingFile r = new RecordingFile(f)) {
				while (r.hasMoreEvents()) {
					RecordedEvent e = r.readEvent();
					if (e.getEventType().getName().equals(eventName))
						n++;
				}
			}
		return n;
	}
}
