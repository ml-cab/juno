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
package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;

/**
 * Saving and restoring session KV through the vision wrapper: a text-only request reaches the text handler; a request
 * whose KV holds an image is refused, since the image belongs to that request and is not in the file.
 */
class VisionAwareForwardPassHandlerSessionTest {

	/** Records the saves and restores that reach it. */
	private static final class SavingStub extends StubForwardPassHandler {
		final List<String> calls = new ArrayList<>();

		@Override
		public long saveKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add("save " + requestId);
			return 10;
		}

		@Override
		public long restoreKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add("restore " + requestId);
			return 10;
		}
	}

	private final SavingStub inner = new SavingStub();
	private final VisionAwareForwardPassHandler handler = new VisionAwareForwardPassHandler(inner, 32000, 64,
			"<image>");
	private final KvSessionStamp stamp = new KvSessionStamp("m", "", "layers-0-2", new int[] { 1, 2 });

	@Test
	@DisplayName("text-only: delegated; a request with an image: refused (REFUSED), until evicted")
	void imageSessionsAreNotSaved() throws Exception {
		Path file = Path.of("unused.kv");
		assertThat(handler.saveKv("text", stamp, file)).isEqualTo(10);
		assertThat(handler.restoreKv("text2", stamp, file)).isEqualTo(10);

		handler.registerVisionEmbeddings("img", new float[][] { new float[64] });
		handler.releaseVisionEmbeddings("img"); // the request finished; its KV still holds the image
		assertThatThrownBy(() -> handler.saveKv("img", stamp, file))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> {
					assertThat(e.reason()).isEqualTo(Reason.REFUSED);
					assertThat(e.getMessage()).contains("image");
				});
		assertThat(inner.calls).containsExactly("save text", "restore text2");

		handler.evict("img");
		assertThat(handler.saveKv("img", stamp, file)).isEqualTo(10);
	}
}
