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
package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.EnumSet;
import java.util.List;

import org.junit.jupiter.api.Test;

import cab.ml.juno.master.ModelLiveChecks.Backend;
import cab.ml.juno.master.ModelLiveChecks.LiveCheck;
import cab.ml.juno.master.ModelLiveChecks.Suite;

class ModelLiveChecksBackendTest {

	@Test
	void autoKeepsEachChecksOwnChoiceAndAnExplicitBackendOverridesIt() {
		assertThat(ModelLiveChecks.useCuda(Backend.AUTO, true)).isTrue();
		assertThat(ModelLiveChecks.useCuda(Backend.AUTO, false)).isFalse();
		assertThat(ModelLiveChecks.useCuda(Backend.CPU, true)).isFalse();
		assertThat(ModelLiveChecks.useCuda(Backend.CUDA, true)).isTrue();
	}

	@Test
	void nodesAreToldTheirBackendOnlyWhenOneWasChosen() {
		assertThat(ModelLiveChecks.nodeUseGpu(Backend.AUTO)).isNull();
		assertThat(ModelLiveChecks.nodeUseGpu(Backend.CPU)).isEqualTo("false");
		assertThat(ModelLiveChecks.nodeUseGpu(Backend.CUDA)).isEqualTo("true");
	}

	@Test
	void cudaRequestedWithoutAUsableGpuFailsEverySelectedCheckClosed() {
		List<LiveCheck> r = ModelLiveChecks.failClosed(EnumSet.of(Suite.TENSOR, Suite.CONTEXT_SHIFT),
				"CUDA requested");
		assertThat(r).extracting(LiveCheck::number).containsExactly(7, 8, 10);
		assertThat(r).allSatisfy(c -> {
			assertThat(c.passed()).isFalse();
			assertThat(c.detail()).contains("CUDA requested");
		});
		assertThat(ModelLiveChecks.failClosed(EnumSet.allOf(Suite.class), "x")).extracting(LiveCheck::number)
				.containsExactly(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	}
}
