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

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Phi-3 file with a 4096-token context and no window key gets a startup notice: Phi-3-mini-4k was trained with
 * a 2047-token window, and a conversion that omits the key runs without it, with no other sign. The notice
 * guesses nothing (the file still gets no window); it only matches that one shape.
 */
@DisplayName("UndeclaredWindowNotice - a Phi-3 4k file without the window key is named at load")
class UndeclaredWindowNoticeTest {

	@TempDir
	Path dir;

	private String notice(Object... keyValues) throws IOException {
		Path gguf = MetadataOnlyGguf.write(dir, "m.gguf", MetadataOnlyGguf.keys(keyValues));
		try (GgufReader r = GgufReader.open(gguf)) {
			return UndeclaredWindowNotice.forFile(r);
		}
	}

	@Test
	@DisplayName("phi3, 4096-token context, no window key: the notice names the key and the trained window")
	void phi3FourKWithoutTheKey() throws IOException {
		String n = notice("general.architecture", "phi3", "phi3.context_length", 4096, "phi3.block_count", 32);
		assertThat(n).isNotNull();
		assertThat(n).contains("phi3.attention.sliding_window").contains("2047").contains("without a window")
				.contains("gguf-info");
		assertThat(n).doesNotContain("\n");
	}

	@Test
	@DisplayName("no notice when the file declares the key, at its width or at 0")
	void declaredKeyGetsNone() throws IOException {
		assertThat(notice("general.architecture", "phi3", "phi3.context_length", 4096,
				"phi3.attention.sliding_window", 2047)).isNull();
		assertThat(notice("general.architecture", "phi3", "phi3.context_length", 4096,
				"phi3.attention.sliding_window", 0)).isNull();
	}

	@Test
	@DisplayName("no notice for other contexts or architectures: Phi-3.5-mini, the 128k models, LLaMA at 4096")
	void otherShapesGetNone() throws IOException {
		assertThat(notice("general.architecture", "phi3", "phi3.context_length", 131072)).isNull();
		assertThat(notice("general.architecture", "phi3")).isNull();
		assertThat(notice("general.architecture", "llama", "llama.context_length", 4096)).isNull();
		assertThat(notice("general.architecture", "phi2", "phi2.context_length", 4096)).isNull();
		assertThat(notice("phi3.context_length", 4096)).isNull();
	}

	@Test
	@DisplayName("the window read at load logs the notice once, as a warning, and still applies no window")
	void forShardLogsItOnce() throws IOException, UnsupportedModelException {
		ShardContext shard = new ShardContext("n0", 0, 2, true, true, 32, 64, 4);

		List<LogRecord> undeclared = new ArrayList<>();
		int[] windows = forShard(shard, undeclared, "general.architecture", "phi3", "phi3.context_length", 4096,
				"phi3.block_count", 2);
		assertThat(windows).containsExactly(0, 0);
		assertThat(undeclared).hasSize(1);
		assertThat(undeclared.get(0).getLevel()).isEqualTo(Level.WARNING);
		assertThat(undeclared.get(0).getMessage()).isEqualTo(notice("general.architecture", "phi3",
				"phi3.context_length", 4096, "phi3.block_count", 2));

		List<LogRecord> declared = new ArrayList<>();
		windows = forShard(shard, declared, "general.architecture", "phi3", "phi3.context_length", 4096,
				"phi3.block_count", 2, "phi3.attention.sliding_window", 2047);
		assertThat(windows).containsExactly(2047, 2047);
		assertThat(declared).noneMatch(r -> r.getLevel() == Level.WARNING);
	}

	private int[] forShard(ShardContext shard, List<LogRecord> into, Object... keyValues)
			throws IOException, UnsupportedModelException {
		Logger log = Logger.getAnonymousLogger();
		log.setUseParentHandlers(false);
		log.setLevel(Level.ALL);
		log.addHandler(new Handler() {
			@Override
			public void publish(LogRecord record) {
				into.add(record);
			}

			@Override
			public void flush() {
			}

			@Override
			public void close() {
			}
		});
		Path gguf = MetadataOnlyGguf.write(dir, "s.gguf", MetadataOnlyGguf.keys(keyValues));
		try (GgufReader r = GgufReader.open(gguf)) {
			return SlidingWindow.forShard(r, shard, log);
		}
	}
}
