package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.google.protobuf.ByteString;

import cab.ml.juno.api.grpc.ActivationDtype;
import cab.ml.juno.coordinator.TensorParallelPipelineClient;
import cab.ml.juno.api.grpc.ForwardRequest;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.LoadShardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.api.grpc.NodeStatusRequest;
import cab.ml.juno.api.grpc.NodeStatusResponse;
import cab.ml.juno.api.grpc.UnloadShardRequest;
import cab.ml.juno.api.grpc.UnloadShardResponse;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * Unit tests for TensorParallelPipelineClient using lightweight in-process gRPC servers.
 *
 * Verifies:
 * 1. forward() dispatches to all N nodes simultaneously (parallel, not serial).
 * 2. Partial logit vectors from all nodes are summed (AllReduce).
 * 3. loadShards() sends tensorRank and tensorWorldSize to every node.
 * 4. All nodes are called in parallel, not sequentially (timing regression).
 */
@DisplayName("TensorParallelPipelineClient — parallel dispatch and AllReduce")
class TensorParallelPipelineClientTest {

    private static final int BASE_PORT  = 29300; // well away from cluster ports (19092-19094)
    private static final int VOCAB_SIZE = 32_000;

    // ── In-process node stub ──────────────────────────────────────────────────

    /**
     * Minimal gRPC server that:
     * - Records received ForwardRequest calls.
     * - Returns a FLOAT32 logit vector with a configurable winner token set to a
     *   given value and all other logits at 0.
     * - Optionally sleeps to simulate per-node latency (for timing tests).
     */
    static class TensorNodeStub {
        final int  port;
        final Server server;
        final CopyOnWriteArrayList<ForwardRequest>   forwardRequests  = new CopyOnWriteArrayList<>();
        final CopyOnWriteArrayList<LoadShardRequest> loadRequests     = new CopyOnWriteArrayList<>();
        final AtomicInteger forwardCount = new AtomicInteger();

        private final int   winnerToken;
        private final float winnerValue;
        private final long  delayMs;

        TensorNodeStub(int port, int winnerToken, float winnerValue, long delayMs) throws Exception {
            this.port        = port;
            this.winnerToken = winnerToken;
            this.winnerValue = winnerValue;
            this.delayMs     = delayMs;

            this.server = ServerBuilder.forPort(port)
                    .addService(new NodeServiceGrpc.NodeServiceImplBase() {

                        @Override
                        public void forwardPass(ForwardRequest request,
                                StreamObserver<ForwardResponse> responseObserver) {
                            forwardRequests.add(request);
                            forwardCount.incrementAndGet();

                            if (delayMs > 0) {
                                try { Thread.sleep(delayMs); } catch (InterruptedException ignored) {}
                            }

                            // Encode a FLOAT32 logit vector with the configured winner
                            float[] logits = new float[VOCAB_SIZE];
                            logits[winnerToken] = winnerValue;
                            byte[] encoded = cab.ml.juno.node.ActivationCodec.encode(
                                    logits, cab.ml.juno.node.ActivationDtype.FLOAT32);

                            responseObserver.onNext(ForwardResponse.newBuilder()
                                    .setRequestId(request.getRequestId())
                                    .setActivation(ByteString.copyFrom(encoded))
                                    .setIsLastNode(true)   // all tensor-parallel nodes are "last"
                                    .setDtype(ActivationDtype.FLOAT32)
                                    .build());
                            responseObserver.onCompleted();
                        }

                        @Override
                        public void loadShard(LoadShardRequest request,
                                StreamObserver<LoadShardResponse> responseObserver) {
                            loadRequests.add(request);
                            responseObserver.onNext(LoadShardResponse.newBuilder()
                                    .setSuccess(true)
                                    .setMessage("ok rank=" + request.getTensorRank()
                                                + " worldSize=" + request.getTensorWorldSize())
                                    .build());
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
                            responseObserver.onNext(NodeStatusResponse.newBuilder()
                                    .setStatus("READY").build());
                            responseObserver.onCompleted();
                        }
                    })
                    .build()
                    .start();
        }

        void shutdown() throws InterruptedException {
            server.shutdown();
            server.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS);
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("forward() dispatches to all 3 tensor-parallel nodes")
    void forward_dispatches_to_all_nodes() throws Exception {
        TensorNodeStub n0 = new TensorNodeStub(BASE_PORT,     42, 100.0f, 0);
        TensorNodeStub n1 = new TensorNodeStub(BASE_PORT + 1, 42, 100.0f, 0);
        TensorNodeStub n2 = new TensorNodeStub(BASE_PORT + 2, 42, 100.0f, 0);

        try {
            TensorParallelPipelineClient client = new TensorParallelPipelineClient(
                    List.of(
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 1),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 2)),
                    VOCAB_SIZE);

            client.forward("req-1", new int[]{1, 2, 3}, 0);

            assertThat(n0.forwardCount.get()).as("node-0 called").isEqualTo(1);
            assertThat(n1.forwardCount.get()).as("node-1 called").isEqualTo(1);
            assertThat(n2.forwardCount.get()).as("node-2 called").isEqualTo(1);

            client.shutdown();
        } finally {
            n0.shutdown(); n1.shutdown(); n2.shutdown();
        }
    }

    @Test
    @DisplayName("forward() sums partial logit vectors from all nodes (AllReduce)")
    void forward_sums_partial_logit_vectors() throws Exception {
        // Each node contributes winnerValue=100.0f at token 42.
        // AllReduce sum of 3 nodes → logits[42] == 300.0f
        TensorNodeStub n0 = new TensorNodeStub(BASE_PORT + 10, 42, 100.0f, 0);
        TensorNodeStub n1 = new TensorNodeStub(BASE_PORT + 11, 42, 100.0f, 0);
        TensorNodeStub n2 = new TensorNodeStub(BASE_PORT + 12, 42, 100.0f, 0);

        try {
            TensorParallelPipelineClient client = new TensorParallelPipelineClient(
                    List.of(
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 10),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 11),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 12)),
                    VOCAB_SIZE);

            float[] logits = client.forward("req-2", new int[]{5}, 0);

            assertThat(logits).hasSize(VOCAB_SIZE);
            assertThat(logits[42])
                    .as("allreduce sum at winner token")
                    .isEqualTo(300.0f);
            // The winner is still identifiable — argmax is preserved
            float max = Float.NEGATIVE_INFINITY;
            for (float v : logits) max = Math.max(max, v);
            assertThat(max).isEqualTo(300.0f);

            client.shutdown();
        } finally {
            n0.shutdown(); n1.shutdown(); n2.shutdown();
        }
    }

    @Test
    @DisplayName("loadShards() sends tensorRank and tensorWorldSize to every node")
    void loadShards_sends_tensor_rank_and_world_size() throws Exception {
        TensorNodeStub n0 = new TensorNodeStub(BASE_PORT + 20, 0, 0f, 0);
        TensorNodeStub n1 = new TensorNodeStub(BASE_PORT + 21, 0, 0f, 0);
        TensorNodeStub n2 = new TensorNodeStub(BASE_PORT + 22, 0, 0f, 0);

        try {
            TensorParallelPipelineClient client = new TensorParallelPipelineClient(
                    List.of(
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 20),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 21),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 22)),
                    VOCAB_SIZE);

            client.loadShards(List.of(
                    new TensorParallelPipelineClient.TensorShardConfig(0, 22, true, true, 0, 3),
                    new TensorParallelPipelineClient.TensorShardConfig(0, 22, true, true, 1, 3),
                    new TensorParallelPipelineClient.TensorShardConfig(0, 22, true, true, 2, 3)));

            assertThat(n0.loadRequests).hasSize(1);
            assertThat(n1.loadRequests).hasSize(1);
            assertThat(n2.loadRequests).hasSize(1);

            assertThat(n0.loadRequests.get(0).getTensorRank()).isEqualTo(0);
            assertThat(n1.loadRequests.get(0).getTensorRank()).isEqualTo(1);
            assertThat(n2.loadRequests.get(0).getTensorRank()).isEqualTo(2);

            for (TensorNodeStub stub : List.of(n0, n1, n2)) {
                assertThat(stub.loadRequests.get(0).getTensorWorldSize())
                        .as("worldSize on " + stub.port)
                        .isEqualTo(3);
                assertThat(stub.loadRequests.get(0).getHasEmbeddings()).isTrue();
                assertThat(stub.loadRequests.get(0).getHasOutputProjection()).isTrue();
            }

            client.shutdown();
        } finally {
            n0.shutdown(); n1.shutdown(); n2.shutdown();
        }
    }

    @Test
    @DisplayName("forward() calls all nodes in parallel, not serially (timing regression)")
    void forward_is_parallel_not_serial() throws Exception {
        // Each node sleeps 200ms. Serial = 600ms. Parallel ≈ 200ms + overhead.
        long delayMs = 200L;
        TensorNodeStub n0 = new TensorNodeStub(BASE_PORT + 30, 0, 1.0f, delayMs);
        TensorNodeStub n1 = new TensorNodeStub(BASE_PORT + 31, 0, 1.0f, delayMs);
        TensorNodeStub n2 = new TensorNodeStub(BASE_PORT + 32, 0, 1.0f, delayMs);

        try {
            TensorParallelPipelineClient client = new TensorParallelPipelineClient(
                    List.of(
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 30),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 31),
                            new TensorParallelPipelineClient.NodeAddress("localhost", BASE_PORT + 32)),
                    VOCAB_SIZE);

            long start   = System.currentTimeMillis();
            client.forward("req-timing", new int[]{1}, 0);
            long elapsed = System.currentTimeMillis() - start;

            // Parallel: should complete in ~1× delay, not 3× delay
            long serialTime = 3 * delayMs; // 600ms
            assertThat(elapsed)
                    .as("parallel dispatch must be faster than serial (" + serialTime + " ms)")
                    .isLessThan(serialTime);

            client.shutdown();
        } finally {
            n0.shutdown(); n1.shutdown(); n2.shutdown();
        }
    }
}