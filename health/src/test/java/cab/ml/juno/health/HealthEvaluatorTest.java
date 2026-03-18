package cab.ml.juno.health;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class HealthEvaluatorTest {

	private HealthEvaluator evaluator;
	private final HealthThresholds thresholds = HealthThresholds.defaults();

	@BeforeEach
	void setUp() {
		evaluator = new HealthEvaluator(thresholds);
	}

	private NodeHealth health(String nodeId, double pressure, long ageMs) {
		Instant sampledAt = Instant.now().minusMillis(ageMs);
		return new NodeHealth(nodeId, pressure, 100_000L, 1_000_000L, 60.0, 50.0, sampledAt);
	}

	private NodeHealth fresh(String nodeId, double pressure) {
		return health(nodeId, pressure, 0);
	}

	@Test
	void no_events_for_healthy_node() {
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.5));
		assertThat(events).isEmpty();
	}

	@Test
	void emits_warning_when_crossing_warning_threshold() {
		evaluator.evaluate(fresh("n1", 0.5)); // baseline OK
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.92));

		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.VRAM_WARNING);
	}

	@Test
	void emits_critical_when_crossing_critical_threshold() {
		evaluator.evaluate(fresh("n1", 0.5));
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.99));

		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.VRAM_CRITICAL);
	}

	@Test
	void no_duplicate_warning_events_on_consecutive_calls() {
		evaluator.evaluate(fresh("n1", 0.5));
		evaluator.evaluate(fresh("n1", 0.92)); // first warning
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.93)); // still warning

		assertThat(events).isEmpty(); // no duplicate
	}

	@Test
	void emits_recovered_when_dropping_below_warning() {
		evaluator.evaluate(fresh("n1", 0.5));
		evaluator.evaluate(fresh("n1", 0.92)); // warning
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.50)); // recovered

		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.NODE_RECOVERED);
	}

	@Test
	void emits_stale_when_health_probe_too_old() {
		// 20s old — exceeds 15s default threshold
		List<HealthEvent> events = evaluator.evaluate(health("n1", 0.5, 20_000));

		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.NODE_STALE);
	}

	@Test
	void emits_recovered_when_stale_node_sends_fresh_probe() {
		evaluator.evaluate(health("n1", 0.5, 20_000)); // stale
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.5)); // fresh

		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.NODE_RECOVERED);
	}

	@Test
	void events_are_independent_per_node() {
		evaluator.evaluate(fresh("n1", 0.92)); // warning for n1
		List<HealthEvent> n2events = evaluator.evaluate(fresh("n2", 0.5)); // n2 healthy

		assertThat(n2events).isEmpty();
	}

	@Test
	void forget_resets_node_state() {
		evaluator.evaluate(fresh("n1", 0.92)); // warning state
		evaluator.forget("n1");

		// After forget, same pressure should re-emit warning
		List<HealthEvent> events = evaluator.evaluate(fresh("n1", 0.92));
		assertThat(events).hasSize(1);
		assertThat(events.get(0).type()).isEqualTo(HealthEvent.EventType.VRAM_WARNING);
	}
}
