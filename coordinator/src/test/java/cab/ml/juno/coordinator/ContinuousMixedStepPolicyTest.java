package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

class ContinuousMixedStepPolicyTest {

	@Test
	void decode_preferred_when_batch_full() {
		List<Member> members = List.of(
				Member.decode("d1"),
				Member.decode("d2"),
				Member.prefill("p1", 10),
				Member.prefill("p2", 10));

		ContinuousMixedStepPolicy.Plan<Member> plan = ContinuousMixedStepPolicy.plan(members, 2, 32,
				Member::isDecode, Member::remainingPrefill);

		assertThat(plan.decode()).extracting(Member::id).containsExactly("d1", "d2");
		assertThat(plan.prefill()).isEmpty();
	}

	@Test
	void leftover_slots_take_prefill_chunks() {
		List<Member> members = List.of(
				Member.decode("d1"),
				Member.prefill("p1", 40),
				Member.prefill("p2", 5));

		ContinuousMixedStepPolicy.Plan<Member> plan = ContinuousMixedStepPolicy.plan(members, 4, 32,
				Member::isDecode, Member::remainingPrefill);

		assertThat(plan.decode()).extracting(Member::id).containsExactly("d1");
		assertThat(plan.prefill()).hasSize(2);
		assertThat(plan.prefill().get(0).member().id()).isEqualTo("p1");
		assertThat(plan.prefill().get(0).tokenBudget()).isEqualTo(32);
		assertThat(plan.prefill().get(1).member().id()).isEqualTo("p2");
		assertThat(plan.prefill().get(1).tokenBudget()).isEqualTo(5);
	}

	@Test
	void empty_running_set_yields_empty_plan() {
		ContinuousMixedStepPolicy.Plan<Member> plan = ContinuousMixedStepPolicy.plan(List.of(), 8, 32,
				Member::isDecode, Member::remainingPrefill);
		assertThat(plan.decode()).isEmpty();
		assertThat(plan.prefill()).isEmpty();
	}

	@Test
	void only_prefill_uses_full_slot_cap() {
		List<Member> members = List.of(
				Member.prefill("p1", 100),
				Member.prefill("p2", 100),
				Member.prefill("p3", 100));

		ContinuousMixedStepPolicy.Plan<Member> plan = ContinuousMixedStepPolicy.plan(members, 2, 16,
				Member::isDecode, Member::remainingPrefill);

		assertThat(plan.decode()).isEmpty();
		assertThat(plan.prefill()).hasSize(2);
		assertThat(plan.prefill().get(0).tokenBudget()).isEqualTo(16);
	}

	@Test
	void chunk_size_one_still_gets_a_slot() {
		List<Member> members = List.of(Member.prefill("p1", 3));
		ContinuousMixedStepPolicy.Plan<Member> plan = ContinuousMixedStepPolicy.plan(members, 1, 1,
				Member::isDecode, Member::remainingPrefill);
		assertThat(plan.prefill()).hasSize(1);
		assertThat(plan.prefill().get(0).tokenBudget()).isEqualTo(1);
	}

	private record Member(String id, boolean decode, int remainingPrefill) {
		static Member decode(String id) {
			return new Member(id, true, 0);
		}

		static Member prefill(String id, int remaining) {
			return new Member(id, false, remaining);
		}

		boolean isDecode() {
			return decode;
		}
	}
}
