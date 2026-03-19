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

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.CpuKVCache;
import cab.ml.juno.kvcache.GpuKVCache;
import cab.ml.juno.kvcache.KVCacheManager;
import cab.ml.juno.kvcache.LayerRange;
import cab.ml.juno.node.ActivationCodec;
import cab.ml.juno.node.CudaAvailability;
import cab.ml.juno.node.CpuForwardPassHandler;
import cab.ml.juno.node.CublasMatVec;
import cab.ml.juno.node.CyclicForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.GpuContext;
import cab.ml.juno.node.GpuForwardPassHandler;
import cab.ml.juno.node.ForwardResult;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.registry.ShardAssignment;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import cab.ml.juno.api.grpc.ActivationDtype;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.LoadShardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.api.grpc.NodeStatusRequest;
import cab.ml.juno.api.grpc.NodeStatusResponse;
import cab.ml.juno.api.grpc.UnloadShardRequest;
import cab.ml.juno.api.grpc.UnloadShardResponse;

/**
 * Minimal gRPC NodeService backed by CyclicForwardPassHandler.
 *
 * Used by ThreeNodeClusterIT — each node JVM runs one of these. No GPU, no real
 * weights — deterministic stub responses only.
 *
 * Activation compression: Each ForwardRequest carries a dtype field that tells
 * this node how to decode the incoming activation bytes. The response
 * activation (for intermediate nodes) is compressed using the same dtype.
 * Final-node logits are always FLOAT32.
 */
public final class EmbeddedNodeServer {

	private static final Logger log = Logger.getLogger(EmbeddedNodeServer.class.getName());

	private final String nodeId;
	private final int port;
	private final Server grpcServer;

	// TinyLlama-1.1B shape constants (used when no model file is supplied)
	public static final int VOCAB_SIZE = 32_000;
	public static final int HIDDEN_DIM = 2_048;
	public static final int NUM_HEADS = 32;
	public static final int TOTAL_LAYERS = 22;

	/** Stub mode (no real model — used by integration tests). */
	public EmbeddedNodeServer(String nodeId, int port) {
		this(nodeId, port, null, true);
	}

	/**
	 * Real-model mode (GPU by default).
	 *
	 * @param modelPath path to a GGUF file, or null to fall back to stub mode.
	 */
	public EmbeddedNodeServer(String nodeId, int port, String modelPath) {
		this(nodeId, port, modelPath, true);
	}

	/**
	 * Real-model mode.
	 *
	 * @param modelPath path to a GGUF file, or null to fall back to stub mode.
	 * @param useGpu   when true use GPU if available; when false use CPU.
	 */
	public EmbeddedNodeServer(String nodeId, int port, String modelPath, boolean useGpu) {
		this.nodeId = nodeId;
		this.port = port;
		this.grpcServer = ServerBuilder.forPort(port).addService(new NodeServiceImpl(nodeId, modelPath, useGpu)).build();
	}

	public void start() throws IOException {
		grpcServer.start();
		log.info("Node [" + nodeId + "] gRPC server started on port " + port);
		Runtime.getRuntime().addShutdownHook(Thread.ofVirtual().unstarted(() -> {
			try {
				stop();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}));
	}

	public void stop() throws InterruptedException {
		if (grpcServer != null)
			grpcServer.shutdown().awaitTermination(5, TimeUnit.SECONDS);
	}

	public void blockUntilShutdown() throws InterruptedException {
		if (grpcServer != null)
			grpcServer.awaitTermination();
	}

	// ── gRPC service impl ─────────────────────────────────────────────────────

	private static final class NodeServiceImpl extends NodeServiceGrpc.NodeServiceImplBase {

		private static final long NODE_VRAM_BUDGET = 512L * 1024 * 1024;

		private final String nodeId;
		private final String modelPath; // null = stub mode
		private final boolean useGpu;
		private volatile ForwardPassHandler handler;
		private volatile ShardContext context;
		private volatile GpuContext gpuContext; // non-null only when handler is GPU
		@SuppressWarnings("unused")
		private volatile KVCacheManager kvCache;

		NodeServiceImpl(String nodeId, String modelPath, boolean useGpu) {
			this.nodeId = nodeId;
			this.modelPath = modelPath;
			this.useGpu = useGpu;
			this.handler = new CyclicForwardPassHandler(); // replaced in loadShard() when real model
			this.context = buildDefaultContext();
			this.kvCache = new KVCacheManager(new GpuKVCache(NODE_VRAM_BUDGET), new CpuKVCache(256));
		}

		@Override
		public void forwardPass(cab.ml.juno.api.grpc.ForwardRequest request,
				StreamObserver<ForwardResponse> responseObserver) {
			try {
				// ── Decode incoming activation ──────────────────────────────
				// Inside forwardPass, after checking request.getError().isEmpty()
				byte[] rawBytes = request.getActivation().toByteArray();
				cab.ml.juno.node.ActivationDtype inDtype = fromProto(request.getDtype());
				cab.ml.juno.node.ForwardRequest nodeReq;

				if (context.hasEmbeddings()) {
					// First node: bytes are raw int32 token IDs
					java.nio.IntBuffer intBuf = java.nio.ByteBuffer.wrap(rawBytes).order(java.nio.ByteOrder.BIG_ENDIAN)
							.asIntBuffer();
					int[] tokenIds = new int[intBuf.remaining()];
					intBuf.get(tokenIds);
					nodeReq = cab.ml.juno.node.ForwardRequest.withTokens(request.getRequestId(), tokenIds,
							request.getSequencePos());
				} else {
					// Subsequent nodes: bytes are compressed activations
					float[] inputActivations = ActivationCodec.decode(rawBytes, inDtype);
					nodeReq = cab.ml.juno.node.ForwardRequest.withActivations(request.getRequestId(),
							inputActivations, request.getSequencePos());
				}

				ForwardResult result = handler.forward(nodeReq, context);

				// ── Encode outgoing activation ──────────────────────────────
				float[] outputFloats = result.isFinalNode() ? result.logits() : result.activations();

				// Final node always returns plain FLOAT32 logits (no loss allowed on vocab)
				cab.ml.juno.node.ActivationDtype outDtype = result.isFinalNode()
						? cab.ml.juno.node.ActivationDtype.FLOAT32
						: inDtype;

				byte[] encodedOutput = ActivationCodec.encode(outputFloats, outDtype);

				ForwardResponse response = ForwardResponse.newBuilder().setRequestId(request.getRequestId())
						.setActivation(com.google.protobuf.ByteString.copyFrom(encodedOutput))
						.setIsLastNode(result.isFinalNode()).setDtype(toProto(outDtype)).build();

				responseObserver.onNext(response);
				responseObserver.onCompleted();

			} catch (Exception e) {
				responseObserver.onNext(ForwardResponse.newBuilder().setRequestId(request.getRequestId())
						.setError(e.getMessage() != null ? e.getMessage() : e.getClass().getName()).build());
				responseObserver.onCompleted();
			}
		}

		@Override
		public void loadShard(LoadShardRequest request, StreamObserver<LoadShardResponse> responseObserver) {
			ShardAssignment assignment = new ShardAssignment(nodeId, "localhost", 0, request.getStartLayer(),
					request.getEndLayer(), request.getHasEmbeddings(), request.getHasOutputProjection());
			ShardContext newCtx = ShardContext.from(assignment, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);

			String msg;
			if (modelPath != null) {
				try {
					if (gpuContext != null) {
						gpuContext.close();
						gpuContext = null;
					}
					log.info("Loading real model from: " + modelPath + " (useGpu=" + useGpu + ")");
					log.info("Shard context: layers " + request.getStartLayer() + "-" + request.getEndLayer()
							+ " embeddings=" + request.getHasEmbeddings() + " outputProj="
							+ request.getHasOutputProjection());
					if (useGpu && CudaAvailability.isAvailable()) {
						gpuContext = GpuContext.init(0);
						CublasMatVec matVec = new CublasMatVec(gpuContext);
						handler = GpuForwardPassHandler.load(Path.of(modelPath), newCtx, matVec);
						msg = "Real shard loaded (GpuForwardPassHandler) layers " + request.getStartLayer() + "–"
								+ request.getEndLayer();
					} else {
						handler = CpuForwardPassHandler.load(Path.of(modelPath), newCtx);
						msg = "Real shard loaded (CpuForwardPassHandler) layers " + request.getStartLayer() + "–"
								+ request.getEndLayer();
					}
					log.info(msg);
				} catch (Exception e) {
					log.severe("FAILED to load real model: " + e.getMessage());
					e.printStackTrace();
					log.warning("Falling back to stub handler");
					if (gpuContext != null) {
						gpuContext.close();
						gpuContext = null;
					}
					handler = new CyclicForwardPassHandler();
					msg = "Stub shard (model load failed: " + e.getMessage() + ") layers " + request.getStartLayer()
							+ "–" + request.getEndLayer();
				}
			} else {
				handler = new CyclicForwardPassHandler();
				msg = "Stub shard loaded layers " + request.getStartLayer() + "–" + request.getEndLayer();
			}
			context = newCtx;

			LayerRange range = LayerRange.of(request.getStartLayer(), request.getEndLayer());
			kvCache = new KVCacheManager(new GpuKVCache(NODE_VRAM_BUDGET), new CpuKVCache(256), range);
			log.info("Node [" + nodeId + "] KVCache scoped to " + range);

			responseObserver.onNext(LoadShardResponse.newBuilder().setSuccess(true).setMessage(msg).build());
			responseObserver.onCompleted();
		}

		@Override
		public void unloadShard(UnloadShardRequest request, StreamObserver<UnloadShardResponse> responseObserver) {
			if (gpuContext != null) {
				gpuContext.close();
				gpuContext = null;
			}
			context = buildDefaultContext();
			responseObserver.onNext(UnloadShardResponse.newBuilder().setSuccess(true).build());
			responseObserver.onCompleted();
		}

		@Override
		public void getNodeStatus(NodeStatusRequest request, StreamObserver<NodeStatusResponse> responseObserver) {
			responseObserver.onNext(NodeStatusResponse.newBuilder().setNodeId(nodeId).setStatus("READY")
					.setVramTotalBytes(4L * 1024 * 1024 * 1024).setVramFreeBytes(3L * 1024 * 1024 * 1024)
					.setSeedScore(1.0).build());
			responseObserver.onCompleted();
		}

		// ── helpers ───────────────────────────────────────────────────────────

		private static ShardContext buildDefaultContext() {
			ShardAssignment full = new ShardAssignment("default", "localhost", 0, 0, TOTAL_LAYERS, true, true);
			return ShardContext.from(full, VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS);
		}

		private static cab.ml.juno.node.ActivationDtype fromProto(ActivationDtype proto) {
			return switch (proto) {
			case FLOAT16 -> cab.ml.juno.node.ActivationDtype.FLOAT16;
			case INT8 -> cab.ml.juno.node.ActivationDtype.INT8;
			default -> cab.ml.juno.node.ActivationDtype.FLOAT32;
			};
		}

		private static ActivationDtype toProto(cab.ml.juno.node.ActivationDtype dtype) {
			return switch (dtype) {
			case FLOAT32 -> ActivationDtype.FLOAT32;
			case FLOAT16 -> ActivationDtype.FLOAT16;
			case INT8 -> ActivationDtype.INT8;
			};
		}
	}
}