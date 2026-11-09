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
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import cab.ml.juno.coordinator.ClusterBatchNotice;
import cab.ml.juno.node.InferencePipeline;

@DisplayName("Cluster - a decode batch is stated at startup")
class ClusterBatchNoticeTest {

	/** What the notice claims: a cluster client gaining a batched decode has to reword it. */
	@ParameterizedTest(name = "{0}")
	@ValueSource(classes = { ProcessPipelineClient.class, TensorParallelPipelineClient.class })
	@DisplayName("every cluster client decodes a batch one request at a time")
	void everyClusterClientDecodesABatchSerially(Class<?> client) throws Exception {
		Method batch = client.getMethod("forwardBatch", List.class, List.class, List.class);

		assertThat(batch.getDeclaringClass())
				.as("%s keeps the one-request-at-a-time default that %s states", client.getSimpleName(),
						ClusterBatchNotice.class.getSimpleName())
				.isEqualTo(InferencePipeline.class);
	}
}
