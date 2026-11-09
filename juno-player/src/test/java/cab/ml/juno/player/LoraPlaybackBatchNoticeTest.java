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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.reflect.Method;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.MultiDecodeForwardRequest;
import cab.ml.juno.node.ShardContext;

@DisplayName("LoRA playback - a decode batch is stated at startup")
class LoraPlaybackBatchNoticeTest {

	@Test
	@DisplayName("applies only to playback with batching on")
	void appliesOnlyToPlaybackWithBatching() {
		assertThat(LoraPlaybackBatchNotice.applies(true, true)).isTrue();
		assertThat(LoraPlaybackBatchNotice.applies(true, false)).isFalse();
		assertThat(LoraPlaybackBatchNotice.applies(false, true)).isFalse();
		assertThat(LoraPlaybackBatchNotice.applies(false, false)).isFalse();
	}

	/** What the notice claims: a handler gaining a batched decode has to reword it. */
	@ParameterizedTest(name = "{0}")
	@ValueSource(strings = { "LoraTrainableHandler", "Qwen2LoraTrainableHandler", "Phi3LoraTrainableHandler",
			"Qwen3LoraTrainableHandler" })
	@DisplayName("every LoRA handler decodes a batch one request at a time, outside the device region")
	void everyLoraHandlerDecodesABatchSerially(String handler) throws Exception {
		Class<?> type = Class.forName("cab.ml.juno.node." + handler);
		Method batch = type.getMethod("forwardMultiDecode", MultiDecodeForwardRequest.class, ShardContext.class);
		Method region = type.getMethod("batchDecodeRegionActive");

		assertThat(batch.getDeclaringClass()).as("%s keeps the one-request-at-a-time default", handler)
				.isEqualTo(ForwardPassHandler.class);
		assertThat(region.getDeclaringClass()).as("%s has no batch region", handler).isEqualTo(ForwardPassHandler.class);
	}
}
