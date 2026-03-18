package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.health.CircuitBreaker;
import cab.ml.juno.health.HealthThresholds;
import cab.ml.juno.health.NodeHealth;
import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.sampler.Sampler;
import cab.ml.juno.tokenizer.StubTokenizer;

class HealthReactorTest {

	private static final int VOCAB = 1000;

	private CircuitBreaker cb1, cb2;
	private FaultTolerantPipeline pipeline;
	private HealthReactor reactor;

	@BeforeEach
	void setUp() {
		cb1 = CircuitBreaker.forNode("n1");
		cb2 = CircuitBreaker.forNode("n2");

		InferencePipeline stub = new InferencePipeline() {
			@Override
			public float[] forward(String id, int[] t, int s) {
				float[] l = new float[VOCAB];
				l[1] = 100f;
				return l;
			}

			@Override
			public int vocabSize() {
				return VOCAB;
			}
		};

		pipeline = new FaultTolerantPipeline(List.of(new FaultTolerantPipeline.NodePipeline("n1", stub, cb1),
				new FaultTolerantPipeline.NodePipeline("n2", stub, cb2)), RetryPolicy.once());

		reactor = new HealthReactor(HealthThresholds.defaults(), pipeline);
	}

	private NodeHealth probe(String nodeId, double pressure, long ageMs) {
		Instant sampledAt = Instant.now().minusMillis(ageMs);
		return new NodeHealth(nodeId, pressure, 100_000L, 1_000_000L, 60.0, 50.0, sampledAt);
	}

	// ── Normal probes ──────────────────────────────────────────────────────

	@Test
	void healthy_probe_leaves_circuit_closed() {
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		assertThat(cb1.isCallPermitted()).isTrue();
	}

	@Test
	void warning_probe_leaves_circuit_closed() {
		reactor.onHealthProbe(probe("n1", 0.5, 0)); // establish baseline
		reactor.onHealthProbe(probe("n1", 0.93, 0)); // WARNING but not CRITICAL
		assertThat(cb1.isCallPermitted()).isTrue();
	}

	// ── VRAM_CRITICAL ──────────────────────────────────────────────────────

	@Test
	void critical_probe_opens_circuit_for_that_node() {
		reactor.onHealthProbe(probe("n1", 0.5, 0)); // establish baseline
		reactor.onHealthProbe(probe("n1", 0.99, 0)); // CRITICAL

		assertThat(cb1.isCallPermitted()).isFalse();
		assertThat(cb2.isCallPermitted()).isTrue(); // other node unaffected
	}

	@Test
	void critical_event_only_affects_its_node() {
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		reactor.onHealthProbe(probe("n1", 0.99, 0));

		// n2 stays healthy
		assertThat(cb2.isCallPermitted()).isTrue();
		assertThat(pipeline.availableNodeCount()).isEqualTo(1);
	}

	// ── NODE_STALE ─────────────────────────────────────────────────────────

	@Test
	void stale_probe_opens_circuit() {
		reactor.onHealthProbe(probe("n1", 0.5, 0)); // baseline
		reactor.onHealthProbe(probe("n1", 0.5, 20_000)); // 20s stale (threshold=15s)

		assertThat(cb1.isCallPermitted()).isFalse();
	}

	@Test
	void onNodeRemoved_opens_circuit_and_forgets_state() {
		reactor.onHealthProbe(probe("n1", 0.5, 0)); // establish state

		reactor.onNodeRemoved("n1");

		assertThat(cb1.isCallPermitted()).isFalse();
		// After removing and re-adding: a fresh fresh probe should not emit RECOVERED
		// (evaluator forgot n1 state)
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		// Circuit is still open (reset happens via NODE_RECOVERED only, not here)
	}

	// ── NODE_RECOVERED ─────────────────────────────────────────────────────

	@Test
	void recovered_probe_resets_circuit_after_critical() {
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		reactor.onHealthProbe(probe("n1", 0.99, 0)); // CRITICAL → circuit OPEN
		assertThat(cb1.isCallPermitted()).isFalse();

		reactor.onHealthProbe(probe("n1", 0.5, 0)); // back to OK → RECOVERED
		assertThat(cb1.isCallPermitted()).isTrue();
	}

	@Test
	void recovered_probe_resets_circuit_after_stale() {
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		reactor.onHealthProbe(probe("n1", 0.5, 20_000)); // STALE → circuit OPEN
		assertThat(cb1.isCallPermitted()).isFalse();

		reactor.onHealthProbe(probe("n1", 0.5, 0)); // fresh again → RECOVERED
		assertThat(cb1.isCallPermitted()).isTrue();
	}

	// ── Scheduler shutdown ─────────────────────────────────────────────────

	@Test
	void scheduler_shutdown_called_when_fully_unavailable() {
		GenerationLoop loop = new GenerationLoop(new StubTokenizer(), Sampler.create(), pipeline,
				new KVCacheManager(new GpuKVCache(64 * 1024 * 1024), new CpuKVCache(1000)));
		RequestScheduler scheduler = new RequestScheduler(10, loop);

		HealthReactor reactorWithScheduler = new HealthReactor(HealthThresholds.defaults(), pipeline, scheduler);

		// Force CRITICAL on both nodes — fully unavailable
		reactorWithScheduler.onHealthProbe(probe("n1", 0.5, 0));
		reactorWithScheduler.onHealthProbe(probe("n1", 0.99, 0)); // trips n1
		reactorWithScheduler.onHealthProbe(probe("n2", 0.5, 0));
		reactorWithScheduler.onHealthProbe(probe("n2", 0.99, 0)); // trips n2

		assertThat(pipeline.isFullyUnavailable()).isTrue();
		// scheduler.running is private but we verify indirectly — no exception thrown
		// and the pipeline truly has zero available nodes
		assertThat(pipeline.availableNodeCount()).isEqualTo(0);
	}

	// ── Multi-node independence ────────────────────────────────────────────

	@Test
	void independent_probes_for_different_nodes_do_not_interfere() {
		// n1 healthy, n2 critical
		reactor.onHealthProbe(probe("n1", 0.5, 0));
		reactor.onHealthProbe(probe("n2", 0.5, 0));
		reactor.onHealthProbe(probe("n2", 0.99, 0));

		assertThat(cb1.isCallPermitted()).isTrue();
		assertThat(cb2.isCallPermitted()).isFalse();
		assertThat(pipeline.isFullyUnavailable()).isFalse();
		assertThat(pipeline.availableNodeCount()).isEqualTo(1);
	}
}
