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
package cab.ml.juno.sampler;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * The floor as the sampler applies it: end-of-sequence is the highest-scoring
 * token here, so a sampler that ignored the floor would return it immediately.
 */
class SamplerMinTokensTest {

	private static final int EOS = 2;

	/** End-of-sequence wins outright unless something holds it back. */
	private static float[] logitsFavouringEos() {
		return new float[] { 1.0f, 2.0f, 20.0f, 3.0f };
	}

	@Test
	void withoutAFloorTheSamplerEndsTheSequenceImmediately() {
		int token = Sampler.create().sample(logitsFavouringEos(), SamplingParams.deterministic(), new int[0], null, null,
				null);

		assertThat(token).isEqualTo(EOS);
	}

	@Test
	void aFloorKeepsTheSequenceGoingUntilItIsReached() {
		Sampler sampler = Sampler.create();
		MinTokenFloor floor = new MinTokenFloor(EOS, 3);
		SamplingParams params = SamplingParams.deterministic();

		for (int generated = 0; generated < 3; generated++) {
			int[] history = new int[generated];
			int token = sampler.sample(logitsFavouringEos(), params, history, null, null, floor);
			assertThat(token).as("token %d of the minimum must not end the sequence", generated + 1).isNotEqualTo(EOS);
		}

		int atFloor = sampler.sample(logitsFavouringEos(), params, new int[3], null, null, floor);
		assertThat(atFloor).as("once the minimum is met the sequence may end").isEqualTo(EOS);
	}

	@Test
	void theFloorPicksTheNextBestTokenRatherThanAnArbitraryOne() {
		int token = Sampler.create().sample(logitsFavouringEos(), SamplingParams.deterministic(), new int[0], null, null,
				new MinTokenFloor(EOS, 5));

		assertThat(token).as("index 3 carries the highest logit after end-of-sequence").isEqualTo(3);
	}

	@Test
	void theFloorCountsFromTheGeneratedHistoryTheSamplerAlreadyReceives() {
		Sampler sampler = Sampler.create();
		MinTokenFloor floor = new MinTokenFloor(EOS, 2);

		assertThat(sampler.sample(logitsFavouringEos(), SamplingParams.deterministic(), new int[] { 7 }, null, null,
				floor)).isNotEqualTo(EOS);
		assertThat(sampler.sample(logitsFavouringEos(), SamplingParams.deterministic(), new int[] { 7, 8 }, null, null,
				floor)).isEqualTo(EOS);
	}

	@Test
	void theCallerLogitsAreNotModified() {
		float[] caller = logitsFavouringEos();
		Sampler.create().sample(caller, SamplingParams.deterministic(), new int[0], null, null, new MinTokenFloor(EOS, 5));

		assertThat(caller).as("the sampler works on a copy, so a masked floor cannot leak back")
				.containsExactly(1.0f, 2.0f, 20.0f, 3.0f);
	}

	@Test
	void minTokensRidesOnTheSamplingParameters() {
		SamplingParams params = SamplingParams.defaults().withMinTokens(16);

		assertThat(params.minTokens()).isEqualTo(16);
		assertThat(SamplingParams.defaults().minTokens()).as("no minimum by default").isZero();
	}

	@Test
	void aNegativeMinimumIsRejectedRatherThanTreatedAsZero() {
		org.assertj.core.api.Assertions.assertThatThrownBy(() -> SamplingParams.defaults().withMinTokens(-1))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
