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
package cab.ml.juno.coordinator;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;

/**
 * Fairness for one continuous engine step: decode slots first; leftover capacity
 * receives prefill ubatch chunks of size {@code --prefill-batch}.
 *
 * <p>Capacity is measured in <em>slots</em> (one decode request or one prefill
 * chunk participant). When the slot budget is exhausted by decode, no prefill
 * chunks are scheduled that step — short decode is not starved by long prompts.
 */
final class ContinuousMixedStepPolicy {

	private ContinuousMixedStepPolicy() {
	}

	record PrefillWork<T>(T member, int tokenBudget) {
	}

	record Plan<T>(List<T> decode, List<PrefillWork<T>> prefill) {
		static <T> Plan<T> empty() {
			return new Plan<>(List.of(), List.of());
		}
	}

	/**
	 * @param members           running-set order (stable selection order)
	 * @param maxSlots          max participants this step ({@code --parallel} cap)
	 * @param prefillChunkSize  {@code --prefill-batch} ubatch size
	 * @param isDecode          true when member is in decode phase
	 * @param remainingPrefill  tokens still to prefill (ignored for decode)
	 */
	static <T> Plan<T> plan(List<T> members, int maxSlots, int prefillChunkSize, Predicate<T> isDecode,
			ToIntFunction<T> remainingPrefill) {
		if (members == null || members.isEmpty() || maxSlots < 1)
			return Plan.empty();

		int chunk = Math.max(1, prefillChunkSize);
		List<T> decode = new ArrayList<>();
		for (T m : members) {
			if (isDecode.test(m))
				decode.add(m);
		}

		List<T> selectedDecode = decode.size() <= maxSlots ? List.copyOf(decode)
				: List.copyOf(decode.subList(0, maxSlots));
		int remainingSlots = maxSlots - selectedDecode.size();
		if (remainingSlots <= 0)
			return new Plan<>(selectedDecode, List.of());

		List<PrefillWork<T>> prefills = new ArrayList<>();
		for (T m : members) {
			if (remainingSlots <= 0)
				break;
			if (isDecode.test(m))
				continue;
			int rem = remainingPrefill.applyAsInt(m);
			if (rem <= 0)
				continue;
			prefills.add(new PrefillWork<>(m, Math.min(chunk, rem)));
			remainingSlots--;
		}
		return new Plan<>(selectedDecode, List.copyOf(prefills));
	}
}
