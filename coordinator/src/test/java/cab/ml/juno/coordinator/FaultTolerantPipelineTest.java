package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.FaultTolerantPipeline;
import cab.ml.juno.coordinator.PipelineUnavailableException;
import cab.ml.juno.coordinator.RetryPolicy;
import cab.ml.juno.health.CircuitBreaker;
import cab.ml.juno.node.InferencePipeline;

class FaultTolerantPipelineTest {

	private static final int VOCAB = 1000;

	// ── Helpers ───────────────────────────────────────────────────────────────

	private InferencePipeline healthy(int winner) {
		return new InferencePipeline() {
			@Override
			public float[] forward(String id, int[] tokens, int start) {
				float[] l = new float[VOCAB];
				l[winner] = 100f;
				return l;
			}

			@Override
			public int vocabSize() {
				return VOCAB;
			}
		};
	}

	private InferencePipeline alwaysFails() {
		return new InferencePipeline() {
			@Override
			public float[] forward(String id, int[] tokens, int start) {
				throw new RuntimeException("simulated node failure");
			}

			@Override
			public int vocabSize() {
				return VOCAB;
			}
		};
	}

	private InferencePipeline failsThenSucceeds(int failCount, int winner) {
		AtomicInteger calls = new AtomicInteger(0);
		return new InferencePipeline() {
			@Override
			public float[] forward(String id, int[] tokens, int start) {
				if (calls.getAndIncrement() < failCount)
					throw new RuntimeException("transient failure #" + calls.get());
				float[] l = new float[VOCAB];
				l[winner] = 100f;
				return l;
			}

			@Override
			public int vocabSize() {
				return VOCAB;
			}
		};
	}

	private FaultTolerantPipeline.NodePipeline node(String id, InferencePipeline p) {
		return FaultTolerantPipeline.NodePipeline.of(id, p);
	}

	private FaultTolerantPipeline.NodePipeline nodeWithCb(String id, InferencePipeline p, CircuitBreaker cb) {
		return new FaultTolerantPipeline.NodePipeline(id, p, cb);
	}

	// ── Basic routing ──────────────────────────────────────────────────────

	@Test
	void routes_to_single_healthy_node() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", healthy(42))), RetryPolicy.none());

		float[] logits = pipeline.forward("req1", new int[] { 1, 2 }, 0);

		assertThat(logits[42]).isGreaterThan(0);
	}

	@Test
	void routes_to_first_healthy_node_when_multiple_exist() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", healthy(10)), node("n2", healthy(20))),
				RetryPolicy.once());

		float[] logits = pipeline.forward("req1", new int[] { 1 }, 0);

		// n1 is first and healthy — should win
		assertThat(logits[10]).isGreaterThan(0);
		assertThat(logits[20]).isEqualTo(0);
	}

	// ── Retry logic ────────────────────────────────────────────────────────

	@Test
	void retries_on_second_node_when_first_fails() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", alwaysFails()), node("n2", healthy(77))),
				RetryPolicy.of(2, 0)); // no backoff in tests

		float[] logits = pipeline.forward("req1", new int[] { 1 }, 0);

		// n1 failed, n2 succeeded
		assertThat(logits[77]).isGreaterThan(0);
	}

	@Test
	void throws_retries_exhausted_when_all_nodes_fail() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", alwaysFails()), node("n2", alwaysFails())),
				RetryPolicy.of(2, 0));

		assertThatThrownBy(() -> pipeline.forward("req1", new int[] { 1 }, 0))
				.isInstanceOf(PipelineUnavailableException.class).satisfies(e -> {
					var ex = (PipelineUnavailableException) e;
					assertThat(ex.reason()).isEqualTo(PipelineUnavailableException.Reason.RETRIES_EXHAUSTED);
					assertThat(ex.attemptsMade()).isEqualTo(2);
					assertThat(ex.isRetryable()).isTrue();
				});
	}

	@Test
	void respects_max_attempts_limit() {
		// 3 nodes, maxAttempts=2 — only first two nodes are tried
		AtomicInteger n3Calls = new AtomicInteger(0);
		var n3 = new InferencePipeline() {
			@Override
			public float[] forward(String id, int[] t, int s) {
				n3Calls.incrementAndGet();
				float[] l = new float[VOCAB];
				l[99] = 100f;
				return l;
			}

			@Override
			public int vocabSize() {
				return VOCAB;
			}
		};

		var pipeline = new FaultTolerantPipeline(
				List.of(node("n1", alwaysFails()), node("n2", alwaysFails()), node("n3", n3)), RetryPolicy.of(2, 0));

		assertThatThrownBy(() -> pipeline.forward("req1", new int[] { 1 }, 0))
				.isInstanceOf(PipelineUnavailableException.class);
		// n3 was never attempted — stopped at maxAttempts=2
		assertThat(n3Calls.get()).isEqualTo(0);
	}

	// ── Circuit breaker integration ────────────────────────────────────────

	@Test
	void records_success_on_healthy_call() {
		CircuitBreaker cb = CircuitBreaker.forNode("n1");
		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), cb)), RetryPolicy.none());

		pipeline.forward("req1", new int[] { 1 }, 0);
		// Circuit should remain CLOSED after a success
		assertThat(cb.isCallPermitted()).isTrue();
	}

	@Test
	void records_failure_and_can_trip_circuit() {
		// Window=1: a single failure fills the window and immediately trips at 50%
		// threshold.
		// (Retry loop iterates over nodes, not attempts — one node means one attempt
		// total.)
		CircuitBreaker cb = new CircuitBreaker("n1", 1, 0.50, Duration.ofSeconds(60));
		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", alwaysFails(), cb)), RetryPolicy.once());

		assertThatThrownBy(() -> pipeline.forward("req1", new int[] { 1 }, 0));

		// One failure fills the window → circuit trips OPEN
		assertThat(cb.isCallPermitted()).isFalse();
	}

	@Test
	void skips_open_circuit_and_uses_next_node() {
		CircuitBreaker openCb = CircuitBreaker.forNode("n1");
		openCb.forceOpen(); // manually trip

		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), openCb), node("n2", healthy(55))),
				RetryPolicy.once());

		float[] logits = pipeline.forward("req1", new int[] { 1 }, 0);

		// n1 was skipped (OPEN), n2 served the call
		assertThat(logits[55]).isGreaterThan(0);
	}

	@Test
	void throws_circuit_open_when_all_circuits_open() {
		CircuitBreaker cb1 = CircuitBreaker.forNode("n1");
		cb1.forceOpen();
		CircuitBreaker cb2 = CircuitBreaker.forNode("n2");
		cb2.forceOpen();

		var pipeline = new FaultTolerantPipeline(
				List.of(nodeWithCb("n1", healthy(1), cb1), nodeWithCb("n2", healthy(2), cb2)), RetryPolicy.once());

		assertThatThrownBy(() -> pipeline.forward("req1", new int[] { 1 }, 0))
				.isInstanceOf(PipelineUnavailableException.class).satisfies(e -> {
					var ex = (PipelineUnavailableException) e;
					assertThat(ex.reason()).isEqualTo(PipelineUnavailableException.Reason.CIRCUIT_OPEN);
					assertThat(ex.attemptsMade()).isEqualTo(0);
					assertThat(ex.isRetryable()).isFalse();
				});
	}

	// ── Health event hooks ─────────────────────────────────────────────────

	@Test
	void onVramCritical_force_opens_that_node_circuit() {
		CircuitBreaker cb = CircuitBreaker.forNode("n1");
		assertThat(cb.isCallPermitted()).isTrue();

		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), cb)), RetryPolicy.none());

		pipeline.onVramCritical("n1");
		assertThat(cb.isCallPermitted()).isFalse();
	}

	@Test
	void onNodeStale_force_opens_that_node_circuit() {
		CircuitBreaker cb = CircuitBreaker.forNode("n1");
		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), cb)), RetryPolicy.none());

		pipeline.onNodeStale("n1");
		assertThat(cb.isCallPermitted()).isFalse();
	}

	@Test
	void onNodeRecovered_resets_circuit_to_closed() {
		CircuitBreaker cb = CircuitBreaker.forNode("n1");
		cb.forceOpen();
		assertThat(cb.isCallPermitted()).isFalse();

		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), cb)), RetryPolicy.none());

		pipeline.onNodeRecovered("n1");
		assertThat(cb.isCallPermitted()).isTrue();
	}

	@Test
	void health_hooks_are_no_ops_for_unknown_node() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", healthy(1))), RetryPolicy.none());

		// Should not throw for an unknown nodeId
		pipeline.onVramCritical("unknown-node");
		pipeline.onNodeStale("unknown-node");
		pipeline.onNodeRecovered("unknown-node");

		// n1 is still healthy
		assertThat(pipeline.availableNodeCount()).isEqualTo(1);
	}

	// ── Availability helpers ───────────────────────────────────────────────

	@Test
	void isFullyUnavailable_true_when_all_circuits_open() {
		CircuitBreaker cb = CircuitBreaker.forNode("n1");
		cb.forceOpen();

		var pipeline = new FaultTolerantPipeline(List.of(nodeWithCb("n1", healthy(1), cb)), RetryPolicy.none());

		assertThat(pipeline.isFullyUnavailable()).isTrue();
		assertThat(pipeline.availableNodeCount()).isEqualTo(0);
	}

	@Test
	void isFullyUnavailable_false_when_at_least_one_circuit_closed() {
		CircuitBreaker cb1 = CircuitBreaker.forNode("n1");
		cb1.forceOpen();
		CircuitBreaker cb2 = CircuitBreaker.forNode("n2"); // CLOSED

		var pipeline = new FaultTolerantPipeline(
				List.of(nodeWithCb("n1", healthy(1), cb1), nodeWithCb("n2", healthy(2), cb2)), RetryPolicy.once());

		assertThat(pipeline.isFullyUnavailable()).isFalse();
		assertThat(pipeline.availableNodeCount()).isEqualTo(1);
	}

	// ── forwardBatch ───────────────────────────────────────────────────────

	@Test
	void forwardBatch_routes_to_first_permitted_node() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", healthy(33)), node("n2", healthy(44))),
				RetryPolicy.once());

		float[][] results = pipeline.forwardBatch(List.of("r1", "r2"), List.of(new int[] { 1 }, new int[] { 2 }),
				List.of(0, 0));

		assertThat(results.length).isEqualTo(2);
		assertThat(results[0][33]).isGreaterThan(0);
		assertThat(results[1][33]).isGreaterThan(0); // both from n1
	}

	@Test
	void forwardBatch_falls_back_to_second_node_on_failure() {
		var pipeline = new FaultTolerantPipeline(List.of(node("n1", alwaysFails()), node("n2", healthy(44))),
				RetryPolicy.of(2, 0));

		float[][] results = pipeline.forwardBatch(List.of("r1"), List.of(new int[] { 1 }), List.of(0));

		assertThat(results[0][44]).isGreaterThan(0);
	}
}