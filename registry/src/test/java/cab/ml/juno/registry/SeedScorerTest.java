package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.time.Instant;

import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.NodeDescriptor;
import cab.ml.juno.registry.NodeStatus;
import cab.ml.juno.registry.SeedScorer;

class SeedScorerTest {

	private final SeedScorer scorer = SeedScorer.defaults();

	private NodeDescriptor node(long vramTotal, long vramFree, NodeStatus status) {
		return new NodeDescriptor("n1", "host", 9092, vramTotal, vramFree, status, 0.0, Instant.now(), Instant.now());
	}

	@Test
	void ready_node_with_full_vram_scores_highest() {
		NodeDescriptor best = node(4_000_000_000L, 4_000_000_000L, NodeStatus.READY);
		double score = scorer.score(best, 1.0);
		assertThat(score).isCloseTo(1.0, within(0.01));
	}

	@Test
	void offline_node_scores_near_zero() {
		NodeDescriptor dead = node(4_000_000_000L, 0L, NodeStatus.OFFLINE);
		double score = scorer.score(dead, 0.0);
		assertThat(score).isCloseTo(0.0, within(0.01));
	}

	@Test
	void high_vram_node_scores_higher_than_low_vram() {
		NodeDescriptor rich = node(8_000_000_000L, 8_000_000_000L, NodeStatus.READY);
		NodeDescriptor poor = node(2_000_000_000L, 1_000_000_000L, NodeStatus.READY);
		assertThat(scorer.score(rich, 0.8)).isGreaterThan(scorer.score(poor, 0.8));
	}

	@Test
	void ready_scores_higher_than_idle_for_same_vram() {
		NodeDescriptor ready = node(4_000_000_000L, 4_000_000_000L, NodeStatus.READY);
		NodeDescriptor idle = node(4_000_000_000L, 4_000_000_000L, NodeStatus.IDLE);
		assertThat(scorer.score(ready, 0.5)).isGreaterThan(scorer.score(idle, 0.5));
	}

	@Test
	void weights_must_sum_to_one() {
		assertThatThrownBy(() -> SeedScorer.withWeights(0.5, 0.5, 0.5)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void score_is_always_between_zero_and_one() {
		NodeDescriptor n = node(4_000_000_000L, 2_000_000_000L, NodeStatus.DEGRADED);
		double score = scorer.score(n, 0.6);
		assertThat(score).isBetween(0.0, 1.0);
	}
}
