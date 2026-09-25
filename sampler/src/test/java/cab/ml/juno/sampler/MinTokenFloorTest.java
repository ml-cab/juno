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
 * The end-of-sequence floor: a request that asks for a minimum number of tokens
 * must not be allowed to end before it reaches that many.
 *
 * <p>The floor works by making the end-of-sequence token unsamplable rather than
 * by ignoring it once sampled. Ignoring it would leave its text in the stream and
 * force every generation path to re-decide what to emit; suppressing it lets the
 * model pick its next-best continuation, which is what a caller asking for more
 * tokens actually wants.
 */
class MinTokenFloorTest {

	private static final int EOS = 2;

	@Test
	void suppressesEndOfSequenceWhileBelowTheFloor() {
		float[] logits = { 1.0f, 2.0f, 9.0f, 3.0f };
		new MinTokenFloor(EOS, 5).mask(logits, 0);

		assertThat(logits[EOS]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(logits[0]).isEqualTo(1.0f);
		assertThat(logits[1]).isEqualTo(2.0f);
		assertThat(logits[3]).isEqualTo(3.0f);
	}

	@Test
	void releasesEndOfSequenceOnceTheFloorIsReached() {
		float[] logits = { 1.0f, 2.0f, 9.0f, 3.0f };
		new MinTokenFloor(EOS, 5).mask(logits, 5);

		assertThat(logits[EOS]).as("the floor is a minimum, not a ban").isEqualTo(9.0f);
	}

	@Test
	void doesNothingWhenNoMinimumWasAskedFor() {
		float[] logits = { 1.0f, 2.0f, 9.0f };
		new MinTokenFloor(EOS, 0).mask(logits, 0);

		assertThat(logits[EOS]).isEqualTo(9.0f);
	}

	@Test
	void leavesEndOfSequenceAloneWhenItIsTheOnlyCandidateLeft() {
		// A grammar can legitimately reduce the legal set to end-of-sequence alone.
		// Masking it then would leave every logit at negative infinity, and the
		// softmax over that is not a distribution — so the floor yields instead of
		// producing a sampler input that cannot be sampled from.
		float[] logits = { Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, 9.0f, Float.NEGATIVE_INFINITY };
		new MinTokenFloor(EOS, 5).mask(logits, 0);

		assertThat(logits[EOS]).as("a request for more tokens must not make sampling impossible")
				.isEqualTo(9.0f);
	}

	@Test
	void stillSuppressesWhenAtLeastOneOtherCandidateSurvives() {
		float[] logits = { Float.NEGATIVE_INFINITY, 0.5f, 9.0f, Float.NEGATIVE_INFINITY };
		new MinTokenFloor(EOS, 5).mask(logits, 0);

		assertThat(logits[EOS]).isEqualTo(Float.NEGATIVE_INFINITY);
		assertThat(logits[1]).isEqualTo(0.5f);
	}

	@Test
	void toleratesAnEndOfSequenceIdOutsideTheVocabulary() {
		float[] logits = { 1.0f, 2.0f };
		new MinTokenFloor(99, 5).mask(logits, 0);

		assertThat(logits).containsExactly(1.0f, 2.0f);
	}

	@Test
	void toleratesMissingLogits() {
		new MinTokenFloor(EOS, 5).mask(null, 0);
		new MinTokenFloor(EOS, 5).mask(new float[0], 0);
	}

	@Test
	void reportsWhetherItIsStillHoldingTheSequenceOpen() {
		MinTokenFloor floor = new MinTokenFloor(EOS, 3);

		assertThat(floor.holdsOpen(0)).isTrue();
		assertThat(floor.holdsOpen(2)).isTrue();
		assertThat(floor.holdsOpen(3)).isFalse();
		assertThat(new MinTokenFloor(EOS, 0).holdsOpen(0)).isFalse();
	}
}
