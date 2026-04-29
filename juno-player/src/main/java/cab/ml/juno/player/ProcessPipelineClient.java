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
 * InferencePipeline that fans a forward pass across N remote node processes
 * over gRPC, in pipeline order, with configurable activation compression.
 *
 * Flow (3-node example): Node-1: ForwardPass(tokens=[...], activation=empty,
 * dtype=X) → encoded activation₁ Node-2: ForwardPass(activation=activation₁,
 * dtype=X) → encoded activation₂ Node-3: ForwardPass(activation=activation₂,
 * dtype=X) → logits (isLastNode=true)
 *
 * Compression: {@code activationDtype} controls how float[] tensors are encoded
 * before each gRPC send and decoded after each gRPC receive. The dtype is
 * carried in the proto message so the receiving node always knows how to
 * decode.
 *
 * FLOAT32 → no compression (baseline, default) FLOAT16 → 2× reduction,
 * negligible accuracy loss (recommended for LAN) INT8 → 4× reduction, ~1%
 * relative error (for bandwidth-constrained nodes)
 *
 * Thread-safe — channels and dtype are immutable after construction.
 */
public final class ProcessPipelineClient implements InferencePipeline {

	private static final Logger log = Logger.getLogger(ProcessPipelineClient.class.getName());

	private final List<NodeStub> stubs;
	private final int vocabSize;
	private final ActivationDtype activationDtype;

	/**
	 * Construct with FLOAT32 (no compression) — preserves backward compatibility.
	 */
	public ProcessPipelineClient(List<NodeAddress> nodes, int vocabSize) {
		this(nodes, vocabSize, ActivationDtype.FLOAT32);
	}

	/**
	 * Construct with an explicit activation dtype.
	 *
	 * @param nodes           addresses of pipeline nodes in order
	 * @param vocabSize       size of the final logit vector
	 * @param activationDtype compression format for activation tensors
	 */
	public ProcessPipelineClient(List<NodeAddress> nodes, int vocabSize, ActivationDtype activationDtype) {
		this.vocabSize = vocabSize;
		this.activationDtype = activationDtype;
		this.stubs = nodes.stream().map(addr -> new NodeStub(addr.host(), addr.port())).toList();
		log.info("ProcessPipelineClient created — dtype=" + activationDtype + ", nodes=" + nodes.size());
	}

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		if (tokens == null)
			throw new IllegalArgumentException("tokenIds cannot be null");
		// First node receives raw token IDs as a byte payload; dtype is set for the
		// response
		byte[] activation = intsToBytes(tokens);

		for (int i = 0; i < stubs.size(); i++) {
			NodeStub stub = stubs.get(i);

			ForwardRequest req = ForwardRequest.newBuilder().setRequestId(requestId).setModelId("stub-model")
					.setSequencePos(startPos).setActivation(ByteString.copyFrom(activation))
					.setDtype(toProto(activationDtype)).build();

			ForwardResponse response = stub.blockingStub.forwardPass(req);

			if (!response.getError().isEmpty()) {
				throw new RuntimeException("Node " + i + " forward pass failed: " + response.getError());
			}

			ActivationDtype responseDtype = fromProto(response.getDtype());
			byte[] rawBytes = response.getActivation().toByteArray();

			if (response.getIsLastNode()) {
				// Final node always returns logits as plain FLOAT32
				return ActivationCodec.decode(rawBytes, ActivationDtype.FLOAT32);
			}

			// Intermediate node: decode the compressed activation, then re-encode for
			// the next hop using the configured dtype (enables per-hop dtype later)
			float[] decoded = ActivationCodec.decode(rawBytes, responseDtype);
			activation = ActivationCodec.encode(decoded, activationDtype);
		}

		throw new IllegalStateException("Pipeline completed without a last-node response");
	}

	@Override
	public int vocabSize() {
		return vocabSize;
	}

	/** Returns the activation dtype this client is configured to use. */
	public ActivationDtype activationDtype() {
		return activationDtype;
	}

	/**
	 * Load a shard assignment onto each node in parallel.
	 *
	 * All N nodes receive their LoadShard RPC concurrently using virtual threads
	 * via CompletableFuture.runAsync(). Total time is bounded by the slowest single
	 * node rather than the sum across all nodes.
	 *
	 * Before this change, loading was sequential: node-1, then node-2, then node-3.
	 * Each node reads the full GGUF file and deserialises its weight shard (~2s
	 * each for TinyLlama-1.1B, ~15s for 7B) — sequential loading tripled startup
	 * time for a 3-node cluster.
	 */
	public void loadShards(List<ShardConfig> shards) {
		if (shards.size() != stubs.size())
			throw new IllegalArgumentException("shards.size() must equal nodes.size()");

		List<CompletableFuture<Void>> futures = new ArrayList<>(stubs.size());

		for (int i = 0; i < stubs.size(); i++) {
			final int idx = i;
			ShardConfig shard = shards.get(idx);
			LoadShardRequest req = LoadShardRequest.newBuilder().setModelId("stub-model")
					.setStartLayer(shard.startLayer()).setEndLayer(shard.endLayer())
					.setHasEmbeddings(shard.hasEmbeddings()).setHasOutputProjection(shard.hasOutputProjection())
					.build();

			futures.add(CompletableFuture.runAsync(() -> {
				LoadShardResponse response = stubs.get(idx).blockingStub.loadShard(req);
				log.info("Node " + idx + " shard load: " + response.getMessage());
			}));
		}

		try {
			CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Shard loading interrupted", e);
		} catch (ExecutionException e) {
			Throwable cause = e.getCause();
			throw new RuntimeException("Shard loading failed on one or more nodes: " + cause.getMessage(), cause);
		}
	}

	/** Shut down all gRPC channels cleanly. */
	public void shutdown() throws InterruptedException {
		for (NodeStub stub : stubs) {
			stub.channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
		}
	}

	// ── Inner types ───────────────────────────────────────────────────────────

	private static final class NodeStub {
		final ManagedChannel channel;
		final NodeServiceGrpc.NodeServiceBlockingStub blockingStub;

		NodeStub(String host, int port) {
			this.channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
			this.blockingStub = NodeServiceGrpc.newBlockingStub(channel);
		}
	}

	public record NodeAddress(String host, int port) {
	}

	public record ShardConfig(int startLayer, int endLayer, boolean hasEmbeddings, boolean hasOutputProjection) {
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private static byte[] intsToBytes(int[] ints) {
		java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(ints.length * 4);
		for (int v : ints)
			buf.putInt(v);
		return buf.array();
	}

	static cab.ml.juno.api.grpc.ActivationDtype toProto(ActivationDtype dtype) {
		return switch (dtype) {
		case FLOAT32 -> cab.ml.juno.api.grpc.ActivationDtype.FLOAT32;
		case FLOAT16 -> cab.ml.juno.api.grpc.ActivationDtype.FLOAT16;
		case INT8 -> cab.ml.juno.api.grpc.ActivationDtype.INT8;
		};
	}

	static ActivationDtype fromProto(cab.ml.juno.api.grpc.ActivationDtype proto) {
		return switch (proto) {
		case FLOAT16 -> ActivationDtype.FLOAT16;
		case INT8 -> ActivationDtype.INT8;
		default -> ActivationDtype.FLOAT32; // FLOAT32 and UNRECOGNIZED
		};
	}
}