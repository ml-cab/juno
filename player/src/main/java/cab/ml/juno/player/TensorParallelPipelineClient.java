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
package cab.ml.juno.player;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import com.google.protobuf.ByteString;

import cab.ml.juno.api.grpc.ForwardRequest;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.LoadShardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.node.ActivationCodec;
import cab.ml.juno.node.ActivationDtype;
import cab.ml.juno.node.InferencePipeline;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

/**
 * Tensor-parallel InferencePipeline that dispatches each decode step to ALL
 * nodes simultaneously and reduces (element-wise sums) the partial results.
 *
 * Pipeline-parallel vs tensor-parallel comparison
 * ───────────────────────────────────────────────── ProcessPipelineClient
 * (pipeline parallel — vertical / depth scaling): Activation flows node-0 →
 * node-1 → node-2 in serial order. Each node computes a contiguous block of
 * transformer layers. Network pattern: N-1 sequential gRPC hops per decode
 * step.
 *
 * TensorParallelPipelineClient (tensor parallel — horizontal / width scaling):
 * The SAME input tokens are sent to ALL nodes simultaneously. Each node
 * computes ALL layers but only its horizontal weight slice (attention heads
 * [headStart, headEnd) and FFN width slice). Each node returns a partial logit
 * vector (same shape as full logits). The coordinator sums all partial vectors
 * (star AllReduce) to produce the full next-token distribution. Network
 * pattern: one broadcast + N parallel gRPC calls per decode step.
 *
 * AllReduce strategy ───────────────── This implementation uses a star
 * AllReduce: the coordinator collects partial results from all N nodes and sums
 * them. This trades coordinator-side CPU work (O(N * vocabSize) additions) for
 * simpler wiring and no inter-node communication — well-suited for commodity
 * LAN clusters without InfiniBand.
 *
 * For a 3-node cluster with vocabSize=32000 each AllReduce is ~96000 float
 * additions — negligible compared to the gRPC round-trip time.
 *
 * Node configuration (set during loadShards) ───────────────── Every
 * tensor-parallel node is loaded with: startLayer = 0, endLayer = totalLayers
 * hasEmbeddings = true, hasOutputProjection = true tensorRank = [0, worldSize),
 * tensorWorldSize = worldSize
 *
 * This means every node independently: - decodes raw token IDs from the
 * activation bytes (hasEmbeddings=true) - returns partial FLOAT32 logits
 * (hasOutputProjection=true)
 *
 * Thread-safe — all fields are immutable after construction.
 */
public final class TensorParallelPipelineClient implements InferencePipeline {

	private static final Logger log = Logger.getLogger(TensorParallelPipelineClient.class.getName());

	private final List<NodeStub> stubs;
	private final int vocabSize;

	/**
	 * Construct a tensor-parallel pipeline client.
	 *
	 * @param nodes     list of node addresses in rank order (index 0 = rank 0)
	 * @param vocabSize vocabulary size — length of each logit vector
	 */
	public TensorParallelPipelineClient(List<NodeAddress> nodes, int vocabSize) {
		if (nodes == null || nodes.isEmpty())
			throw new IllegalArgumentException("nodes must not be empty");
		if (vocabSize < 1)
			throw new IllegalArgumentException("vocabSize must be >= 1");

		this.vocabSize = vocabSize;
		this.stubs = nodes.stream().map(addr -> new NodeStub(addr.host(), addr.port())).toList();

		log.info("TensorParallelPipelineClient created — worldSize=" + nodes.size() + ", vocabSize=" + vocabSize);
	}

	// ── InferencePipeline ─────────────────────────────────────────────────────

	/**
	 * Broadcast the token sequence to all tensor-parallel nodes in parallel, then
	 * reduce (element-wise sum) the partial logit vectors they return.
	 *
	 * Step 1: encode token IDs as raw int32 bytes (big-endian). Step 2: dispatch
	 * ForwardRequest to every node concurrently via virtual threads. Step 3: wait
	 * for all nodes to respond (CompletableFuture.allOf). Step 4: element-wise sum
	 * of all partial float[] logit arrays (AllReduce). Step 5: return the summed
	 * logit array to GenerationLoop for sampling.
	 *
	 * The winner token is determined by the sum: if node-0 puts 100.0 on token 42
	 * and node-1 puts 100.0 on token 42, the sum is 200.0 — the argmax is
	 * preserved. In real weight-sliced inference each node produces a scaled
	 * partial contribution that sums to the correct full logit.
	 */
	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		if (tokens == null)
			throw new IllegalArgumentException("tokens cannot be null");

		byte[] tokenBytes = intsToBytes(tokens);

		// Dispatch all nodes in parallel
		List<CompletableFuture<float[]>> futures = new ArrayList<>(stubs.size());
		for (NodeStub stub : stubs) {
			futures.add(CompletableFuture.supplyAsync(() -> {
				ForwardRequest req = ForwardRequest.newBuilder().setRequestId(requestId).setModelId("stub-model")
						.setSequencePos(startPos).setActivation(ByteString.copyFrom(tokenBytes))
						.setDtype(cab.ml.juno.api.grpc.ActivationDtype.FLOAT32).build();

				ForwardResponse response = stub.blockingStub.forwardPass(req);

				if (!response.getError().isEmpty()) {
					throw new RuntimeException(
							"Tensor-parallel forward pass failed on node " + stub.address + ": " + response.getError());
				}

				return ActivationCodec.decode(response.getActivation().toByteArray(), ActivationDtype.FLOAT32);
			}));
		}

		// AllReduce — element-wise sum of all partial logit vectors
		return allReduceSum(futures, vocabSize);
	}

	@Override
	public int vocabSize() {
		return vocabSize;
	}

	// ── Shard loading ─────────────────────────────────────────────────────────

	/**
	 * Load a tensor-parallel shard onto each node in parallel.
	 *
	 * Every node receives the full layer range [startLayer, endLayer] plus its
	 * unique tensorRank and the shared tensorWorldSize. The node uses its rank to
	 * slice the attention head and FFN weight matrices at inference time.
	 *
	 * Loading is fully parallel — total time is bounded by the slowest node.
	 */
	public void loadShards(List<TensorShardConfig> shards) {
		if (shards.size() != stubs.size())
			throw new IllegalArgumentException(
					"shards.size() (" + shards.size() + ") must equal nodes.size() (" + stubs.size() + ")");

		List<CompletableFuture<Void>> futures = new ArrayList<>(stubs.size());

		for (int i = 0; i < stubs.size(); i++) {
			final int idx = i;
			TensorShardConfig shard = shards.get(idx);

			LoadShardRequest req = LoadShardRequest.newBuilder().setModelId("stub-model")
					.setStartLayer(shard.startLayer()).setEndLayer(shard.endLayer())
					.setHasEmbeddings(shard.hasEmbeddings()).setHasOutputProjection(shard.hasOutputProjection())
					.setTensorRank(shard.tensorRank()).setTensorWorldSize(shard.tensorWorldSize()).build();

			futures.add(CompletableFuture.runAsync(() -> {
				LoadShardResponse response = stubs.get(idx).blockingStub.loadShard(req);
				log.info("Tensor-parallel node " + idx + " (rank=" + shard.tensorRank() + ") shard load: "
						+ response.getMessage());
			}));
		}

		awaitAll(futures, "Tensor-parallel shard loading");
	}

	/** Shut down all gRPC channels cleanly. */
	public void shutdown() throws InterruptedException {
		for (NodeStub stub : stubs) {
			stub.channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
		}
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private static final class NodeStub {
		final String address;
		final ManagedChannel channel;
		final NodeServiceGrpc.NodeServiceBlockingStub blockingStub;

		NodeStub(String host, int port) {
			this.address = host + ":" + port;
			this.channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
			this.blockingStub = NodeServiceGrpc.newBlockingStub(channel);
		}
	}

	/**
	 * Tensor-parallel shard configuration sent to each node during loadShards.
	 *
	 * In tensor-parallel mode all nodes receive: startLayer = 0, endLayer =
	 * totalLayers hasEmbeddings = true, hasOutputProjection = true tensorRank = [0,
	 * tensorWorldSize) tensorWorldSize = number of tensor-parallel nodes
	 */
	public record TensorShardConfig(int startLayer, int endLayer, boolean hasEmbeddings, boolean hasOutputProjection,
			int tensorRank, int tensorWorldSize) {
	}

	public record NodeAddress(String host, int port) {
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	/**
	 * Encode a token-ID array as big-endian int32 bytes (same format as
	 * ProcessPipelineClient).
	 */
	private static byte[] intsToBytes(int[] ints) {
		java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(ints.length * 4);
		for (int v : ints)
			buf.putInt(v);
		return buf.array();
	}

	/**
	 * Element-wise sum of partial float[] results collected from all node futures
	 * (star AllReduce). Each future represents one tensor-parallel node's partial
	 * logit vector.
	 */
	private static float[] allReduceSum(List<CompletableFuture<float[]>> futures, int vectorLen) {
		float[] result = new float[vectorLen];
		try {
			CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Tensor-parallel forward pass interrupted", e);
		} catch (ExecutionException e) {
			Throwable cause = e.getCause();
			throw new RuntimeException("Tensor-parallel forward pass failed: " + cause.getMessage(), cause);
		}

		for (CompletableFuture<float[]> future : futures) {
			float[] partial;
			try {
				partial = future.get();
			} catch (InterruptedException | ExecutionException e) {
				throw new RuntimeException("Unexpected error reading completed future", e);
			}
			if (partial.length != vectorLen) {
				throw new IllegalStateException(
						"Partial logit vector length " + partial.length + " != expected vocabSize " + vectorLen);
			}
			for (int i = 0; i < vectorLen; i++) {
				result[i] += partial[i];
			}
		}
		return result;
	}

	private static void awaitAll(List<CompletableFuture<Void>> futures, String context) {
		try {
			CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException(context + " interrupted", e);
		} catch (ExecutionException e) {
			Throwable cause = e.getCause();
			throw new RuntimeException(context + " failed: " + cause.getMessage(), cause);
		}
	}
}