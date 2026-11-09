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
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** A handler outside the device-resident decode region says so once, and says its batched decode stays host-driven. */
@DisplayName("Decode batch region - fail-closed announcement")
class BatchDecodeAnnouncementTest {

	@Test
	@DisplayName("the handlers outside the region announce the host-driven batch once; the handlers in it stay quiet")
	void announcesOnce() throws Exception {
		assumeTrue(GpuResidencyOptions.fromEnv().requested(), "the region is only announced when requested");
		forgetEarlierAnnouncement("the qwen3moe handler");
		Logger log = Logger.getLogger("batch-decode-announcement-" + System.nanoTime());
		log.setUseParentHandlers(false);
		List<LogRecord> records = new ArrayList<>();
		log.addHandler(new Handler() {
			@Override
			public void publish(LogRecord r) {
				records.add(r);
			}

			@Override
			public void flush() {
			}

			@Override
			public void close() {
			}
		});
		log.setLevel(Level.ALL);

		ForwardPassHandlerLoader.announceNoDecodeRegion(log, "qwen3moe");
		ForwardPassHandlerLoader.announceNoDecodeRegion(log, "qwen3moe");
		ForwardPassHandlerLoader.announceNoDecodeRegion(log, "qwen3");
		ForwardPassHandlerLoader.announceNoDecodeRegion(log, "llama");

		assertThat(records).as("one announcement, for the handler outside the region").hasSize(1);
		assertThat(records.get(0).getMessage()).contains("qwen3moe").contains("batch").contains("host-driven");
	}

	/** The announcement is once per JVM and surface; an earlier test in this JVM may already have made this one. */
	@SuppressWarnings("unchecked")
	private static void forgetEarlierAnnouncement(String surface) throws Exception {
		Field announced = GpuResidencyOptions.class.getDeclaredField("ANNOUNCED");
		announced.setAccessible(true);
		((Set<String>) announced.get(null)).remove(surface);
	}
}
