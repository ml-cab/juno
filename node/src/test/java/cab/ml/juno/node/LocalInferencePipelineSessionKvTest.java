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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

@DisplayName("LocalInferencePipeline - saving and restoring a session's KV on every stage")
class LocalInferencePipelineSessionKvTest {

	/** Records what reaches it; fails its restore when told to. */
	private static final class Recording implements ForwardPassHandler {
		final List<String> calls;
		final String name;
		boolean failRestore;

		Recording(String name, List<String> calls) {
			this.name = name;
			this.calls = calls;
		}

		@Override
		public ForwardResult forward(ForwardRequest request, ShardContext context) {
			throw new UnsupportedOperationException();
		}

		@Override
		public boolean isReady() {
			return true;
		}

		@Override
		public long saveKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add(name + " save " + requestId + " " + stamp.stage() + " " + file.getFileName());
			return 100;
		}

		@Override
		public long restoreKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add(name + " restore " + requestId + " " + stamp.stage() + " " + file.getFileName());
			if (failRestore)
				throw new KvSessionFileException(Reason.MISMATCH, name + " differs");
			return 100;
		}

		@Override
		public void evict(String requestId) {
			calls.add(name + " evict " + requestId);
		}
	}

	private static ShardMap nodes(int n) {
		List<ShardAssignment> a = new ArrayList<>();
		for (int i = 0; i < n; i++)
			a.add(new ShardAssignment("n" + i, "host", 9091 + i, 2 * i, 2 * i + 2, i == 0, i == n - 1));
		return new ShardMap("tiny", 2 * n, a, Instant.now());
	}

	private final KvSessionStamp stamp = new KvSessionStamp("m", "", "all", new int[] { 1, 2, 3 });
	private final Path dir = Path.of("/sessions");

	@Test
	@DisplayName("each distinct handler once, its file named for the session and the layers it holds, in stage order")
	void savesOneFilePerHandler() throws Exception {
		List<String> calls = new ArrayList<>();
		Recording a = new Recording("a", calls), b = new Recording("b", calls);
		LocalInferencePipeline p = LocalInferencePipeline.from(nodes(3), List.of(a, b, a), 16, 8, 2);
		List<SessionKvPart> parts = p.saveKv("chat-1", stamp, dir);
		assertThat(calls).containsExactly("a save chat-1 layers-0-6 chat-1.layers-0-6.kv",
				"b save chat-1 layers-2-4 chat-1.layers-2-4.kv");
		assertThat(parts).containsExactly(new SessionKvPart("layers-0-6", "chat-1.layers-0-6.kv", 100),
				new SessionKvPart("layers-2-4", "chat-1.layers-2-4.kv", 100));
	}

	@Test
	@DisplayName("a restore refused on any stage leaves the session without KV on every stage")
	void failedRestoreEvictsEverywhere() {
		List<String> calls = new ArrayList<>();
		Recording a = new Recording("a", calls), b = new Recording("b", calls);
		b.failRestore = true;
		LocalInferencePipeline p = LocalInferencePipeline.from(nodes(2), List.of(a, b), 16, 8, 2);
		assertThatThrownBy(() -> p.restoreKv("chat-1", stamp, dir))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> assertThat(e.reason()).isEqualTo(Reason.MISMATCH));
		assertThat(calls).containsExactly("a restore chat-1 layers-0-2 chat-1.layers-0-2.kv",
				"b restore chat-1 layers-2-4 chat-1.layers-2-4.kv", "a evict chat-1", "b evict chat-1");
	}

	@Test
	@DisplayName("dropping a session's KV reaches every stage")
	void dropReachesEveryStage() {
		List<String> calls = new ArrayList<>();
		LocalInferencePipeline p = LocalInferencePipeline.from(nodes(2),
				List.of(new Recording("a", calls), new Recording("b", calls)), 16, 8, 2);
		p.dropKv("chat-1");
		assertThat(calls).containsExactly("a evict chat-1", "b evict chat-1");
	}
}
