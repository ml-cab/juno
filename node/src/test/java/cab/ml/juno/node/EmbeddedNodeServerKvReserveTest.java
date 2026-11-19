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
package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.ServerSocket;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.google.protobuf.ByteString;

import cab.ml.juno.api.grpc.ActivationDtype;
import cab.ml.juno.api.grpc.ForwardRequest;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

/**
 * A cluster node gives its handler the KV reservation a forward request carries ({@code kv_reserve_positions}), before
 * that request's forward pass, and only when the field is set, so a remote stage's dense KV grows toward the request's
 * length as a local stage's does.
 */
@DisplayName("EmbeddedNodeServer applies a forward request's KV reservation")
class EmbeddedNodeServerKvReserveTest {

	@TempDir
	Path dir;

	private EmbeddedNodeServer server;
	private ManagedChannel channel;

	@AfterEach
	void tearDown() throws Exception {
		if (channel != null)
			channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
		if (server != null)
			server.stop();
	}

	/** Records the calls a node makes on its handler, in order. */
	private static final class RecordingHandler implements ForwardPassHandler {
		final List<String> calls = new CopyOnWriteArrayList<>();

		@Override
		public ForwardResult forward(cab.ml.juno.node.ForwardRequest request, ShardContext context) {
			calls.add("forward " + request.requestId());
			return ForwardResult.logits(request.requestId(), new float[context.vocabSize()], 0);
		}

		@Override
		public void reserveKv(String requestId, int positions) {
			calls.add("reserve " + requestId + " " + positions);
		}

		@Override
		public boolean isReady() {
			return true;
		}
	}

	@Test
	@DisplayName("the reservation reaches the handler before the forward pass that carries it, and only that one")
	void reservationPrecedesItsForwardPass() throws Exception {
		RecordingHandler handler = new RecordingHandler();
		Path gguf = MetadataOnlyGguf.write(dir, "llama");
		int port = freePort();
		server = new EmbeddedNodeServer("n0", port, gguf.toString(), false, (model, ctx, backend, adapters) -> handler);
		server.start();
		channel = ManagedChannelBuilder.forAddress("localhost", port).usePlaintext().build();
		var stub = NodeServiceGrpc.newBlockingStub(channel).withDeadlineAfter(60, TimeUnit.SECONDS);

		assertThat(stub.loadShard(LoadShardRequest.newBuilder().setModelId("m").setStartLayer(0).setEndLayer(4)
				.setHasEmbeddings(true).setHasOutputProjection(true).build()).getSuccess()).isTrue();

		ForwardResponse first = stub.forwardPass(forward("r1", 0, 2112));
		ForwardResponse second = stub.forwardPass(forward("r1", 1, 0));

		assertThat(first.getError()).isEmpty();
		assertThat(second.getError()).isEmpty();
		assertThat(handler.calls).containsExactly("reserve r1 2112", "forward r1", "forward r1");
	}

	private static ForwardRequest forward(String requestId, int pos, int reserve) {
		byte[] tokens = java.nio.ByteBuffer.allocate(4).putInt(1).array();
		return ForwardRequest.newBuilder().setRequestId(requestId).setModelId("m").setSequencePos(pos)
				.setActivation(ByteString.copyFrom(tokens)).setDtype(ActivationDtype.FLOAT32)
				.setKvReservePositions(reserve).build();
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
