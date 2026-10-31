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
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Collects the engine's log messages in-process. Juno reports which device paths it took (the resident
 * decode region, the tiled K-quant kernel, the packed weights) only as log lines, which the smoke scripts
 * grep; a probe reads the same lines here.
 */
final class NoticeCapture extends Handler implements AutoCloseable {

	private final List<String> messages = new ArrayList<>();

	private NoticeCapture() {
		setLevel(Level.INFO);
	}

	static NoticeCapture install() {
		NoticeCapture c = new NoticeCapture();
		Logger.getLogger("").addHandler(c);
		return c;
	}

	@Override
	public synchronized void publish(LogRecord r) {
		if (r.getLevel().intValue() >= Level.INFO.intValue() && r.getMessage() != null)
			messages.add(r.getMessage());
	}

	synchronized List<String> messages() {
		return List.copyOf(messages);
	}

	/** The first message containing {@code text}, or {@code null}. */
	synchronized String first(String text) {
		for (String m : messages)
			if (m.contains(text))
				return m;
		return null;
	}

	@Override
	public void flush() {
	}

	@Override
	public void close() {
		Logger.getLogger("").removeHandler(this);
	}
}
