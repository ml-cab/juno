package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.api.grpc.ForwardRequest;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.LoadShardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.api.grpc.NodeStatusRequest;
import cab.ml.juno.api.grpc.NodeStatusResponse;
import cab.ml.juno.api.grpc.UnloadShardRequest;
import cab.ml.juno.api.grpc.UnloadShardResponse;
import cab.ml.juno.node.ActivationDtype;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * Unit tests for ProcessPipelineClient.loadShards().
 *
 * Spins up lightweight in-process gRPC servers (no real forked JVMs) to verify:
 * 1. All N nodes receive a LoadShard RPC. 2. loadShards() is parallel — nodes
 * are contacted concurrently, not serially. Measured by checking that a
 * shard-load with a deliberate per-node delay completes in roughly (1 × delay)
 * rather than (N × delay).
 */
@DisplayName("ProcessPipelineClient — parallel shard loading")
class LoadShardsParallelTest {

	private static final int BASE_PORT = 29200; // well away from cluster ports

	// ── Minimal in-process gRPC server ───────────────────────────────────────

	/**
	 * Tiny gRPC server that records received LoadShard calls and optionally sleeps
	 * to simulate I/O latency.
	 */
	static class TrackingNodeServer {
		final int port;
		final Server server;
		final CopyOnWriteArrayList<LoadShardRequest> received = new CopyOnWriteArrayList<>();
		final AtomicInteger loadCount = new AtomicInteger();

		// How long each loadShard call blocks (ms)
		@SuppressWarnings("unused")
		private final long delayMs;

		TrackingNodeServer(int port, long delayMs) throws Exception {
			this.port = port;
			this.delayMs = delayMs;
			this.server = ServerBuilder.forPort(port).addService(new NodeServiceGrpc.NodeServiceImplBase() {
				@Override
				public void loadShard(LoadShardRequest request, StreamObserver<LoadShardResponse> responseObserver) {
					received.add(request);
					loadCount.incrementAndGet();
					if (delayMs > 0) {
						try {
							Thread.sleep(delayMs);
						} catch (InterruptedException ignored) {
						}
					}
					responseObserver.onNext(LoadShardResponse.newBuilder()
							.setMessage("ok layers " + request.getStartLayer() + "-" + request.getEndLayer()).build());
					responseObserver.onCompleted();
				}

				@Override
				public void forwardPass(ForwardRequest request, StreamObserver<ForwardResponse> responseObserver) {
					responseObserver.onNext(ForwardResponse.newBuilder().build());
					responseObserver.onCompleted();
				}

				@Override
				public void unloadShard(UnloadShardRequest request,
						StreamObserver<UnloadShardResponse> responseObserver) {
					responseObserver.onNext(UnloadShardResponse.newBuilder().build());
					responseObserver.onCompleted();
				}

				@Override
				public void getNodeStatus(NodeStatusRequest request,
						StreamObserver<NodeStatusResponse> responseObserver) {
					responseObserver.onNext(NodeStatusResponse.newBuilder().build());
					responseObserver.onCompleted();
				}
			}).build().start();
		}

		void stop() {
			server.shutdownNow();
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private List<ProcessPipelineClient.ShardConfig> threeNodeShards() {
		return List.of(new ProcessPipelineClient.ShardConfig(0, 8, true, false),
				new ProcessPipelineClient.ShardConfig(8, 15, false, false),
				new ProcessPipelineClient.ShardConfig(15, 22, false, true));
	}

	// ── Test 1 ────────────────────────────────────────────────────────────────

	@Test
	@DisplayName("All 3 nodes each receive exactly one LoadShard RPC")
	void all_nodes_receive_load_shard() throws Exception {
		int n = 3;
		List<TrackingNodeServer> servers = new ArrayList<>();
		for (int i = 0; i < n; i++)
			servers.add(new TrackingNodeServer(BASE_PORT + i, 0));

		List<ProcessPipelineClient.NodeAddress> addrs = new ArrayList<>();
		for (int i = 0; i < n; i++)
			addrs.add(new ProcessPipelineClient.NodeAddress("localhost", BASE_PORT + i));

		ProcessPipelineClient client = new ProcessPipelineClient(addrs, 32000, ActivationDtype.FLOAT32);
		try {
			client.loadShards(threeNodeShards());

			for (int i = 0; i < n; i++) {
				assertThat(servers.get(i).loadCount.get()).as("node %d should have received 1 LoadShard call", i)
						.isEqualTo(1);
			}
			// Verify shard assignments delivered correctly
			assertThat(servers.get(0).received.get(0).getHasEmbeddings()).isTrue();
			assertThat(servers.get(2).received.get(0).getHasOutputProjection()).isTrue();

		} finally {
			client.shutdown();
			servers.forEach(TrackingNodeServer::stop);
		}
	}

	// ── Test 2 ────────────────────────────────────────────────────────────────

	/**
	 * Parallelism test: 3 nodes each sleep 300ms in loadShard.
	 *
	 * Sequential loading: 3 × 300ms = ~900ms. Parallel loading: 1 × 300ms = ~300ms
	 * (+ overhead).
	 *
	 * We assert total elapsed < 600ms — well below the sequential lower bound. This
	 * test FAILS before the parallel fix and PASSES after.
	 */
	@Test
	@DisplayName("loadShards fires all nodes concurrently (completes in ~1× delay, not 3×)")
	void load_shards_is_parallel_not_serial() throws Exception {
		long delayMs = 300L;
		int n = 3;

		List<TrackingNodeServer> servers = new ArrayList<>();
		for (int i = 0; i < n; i++)
			servers.add(new TrackingNodeServer(BASE_PORT + 10 + i, delayMs));

		List<ProcessPipelineClient.NodeAddress> addrs = new ArrayList<>();
		for (int i = 0; i < n; i++)
			addrs.add(new ProcessPipelineClient.NodeAddress("localhost", BASE_PORT + 10 + i));

		ProcessPipelineClient client = new ProcessPipelineClient(addrs, 32000, ActivationDtype.FLOAT32);
		try {
			long start = System.currentTimeMillis();
			client.loadShards(threeNodeShards());
			long elapsed = System.currentTimeMillis() - start;

			// All 3 nodes served
			for (int i = 0; i < n; i++)
				assertThat(servers.get(i).loadCount.get()).isEqualTo(1);

			// Must complete well under serial time (3 × 300ms = 900ms).
			// Parallel: ~300ms + overhead. Threshold is generous at 600ms.
			assertThat(elapsed).as("loadShards must be parallel: expected ~%dms, got %dms", delayMs, elapsed)
					.isLessThan(delayMs * 2);

		} finally {
			client.shutdown();
			servers.forEach(TrackingNodeServer::stop);
		}
	}
}