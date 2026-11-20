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

import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.api.grpc.SessionKvRequest;
import cab.ml.juno.api.grpc.SessionKvResponse;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.kvcache.SessionDirOptions;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

/**
 * A cluster node saves and restores its own layers of a session under its own session directory, labelled by its layer
 * range (and its rank on a tensor-parallel node), passes a refusal back as its reason, refuses without a directory, and
 * frees a dropped session's KV.
 */
@DisplayName("EmbeddedNodeServer - a node's layers of a session saved, restored and dropped")
class EmbeddedNodeServerSessionKvTest {

	@TempDir
	Path dir;

	private final String prevDir = System.getProperty(SessionDirOptions.ENV);
	private EmbeddedNodeServer server;
	private ManagedChannel channel;

	@AfterEach
	void tearDown() throws Exception {
		if (prevDir == null)
			System.clearProperty(SessionDirOptions.ENV);
		else
			System.setProperty(SessionDirOptions.ENV, prevDir);
		if (channel != null)
			channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
		if (server != null)
			server.stop();
	}

	/** Records the session calls that reach it; refuses a restore of "bad". */
	private static final class RecordingHandler implements ForwardPassHandler {
		final List<String> calls = new CopyOnWriteArrayList<>();

		@Override
		public ForwardResult forward(cab.ml.juno.node.ForwardRequest request, ShardContext context) {
			return ForwardResult.logits(request.requestId(), new float[context.vocabSize()], 0);
		}

		@Override
		public long saveKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add("save " + requestId + " " + stamp + " " + file);
			return 42;
		}

		@Override
		public long restoreKv(String requestId, KvSessionStamp stamp, Path file) {
			calls.add("restore " + requestId + " " + file.getFileName());
			if (requestId.equals("bad"))
				throw new KvSessionFileException(Reason.MISMATCH, "model differs");
			return 42;
		}

		@Override
		public void evict(String requestId) {
			calls.add("evict " + requestId);
		}

		@Override
		public boolean isReady() {
			return true;
		}
	}

	@Test
	@DisplayName("pipeline stage: files under the node's directory named for its layers; refusals and drops pass through")
	void pipelineStage() throws Exception {
		RecordingHandler handler = new RecordingHandler();
		var stub = start(handler, 4, 8, 0, 1);
		Path sessions = dir.resolve("sessions");
		System.setProperty(SessionDirOptions.ENV, sessions.toString());

		SessionKvResponse saved = stub.saveSessionKv(request("chat-1"));
		assertThat(saved.getErrorReason()).isEmpty();
		assertThat(saved.getStage()).isEqualTo("layers-4-8");
		assertThat(saved.getFile()).isEqualTo("chat-1.layers-4-8.kv");
		assertThat(saved.getBytes()).isEqualTo(42);
		assertThat(handler.calls.get(0)).isEqualTo("save chat-1 KvSessionStamp[model=m.gguf, adapter=a.lora, stage="
				+ "layers-4-8, positions=3] " + sessions.resolve("chat-1.layers-4-8.kv"));
		assertThat(sessions).isDirectory();

		SessionKvResponse restored = stub.restoreSessionKv(request("chat-1"));
		assertThat(restored.getErrorReason()).isEmpty();
		SessionKvResponse refused = stub.restoreSessionKv(request("bad"));
		assertThat(refused.getErrorReason()).isEqualTo("MISMATCH");
		assertThat(refused.getError()).contains("model differs");

		assertThat(stub.dropSessionKv(SessionKvRequest.newBuilder().setSessionId("chat-1").build()).getErrorReason())
				.isEmpty();
		assertThat(handler.calls).containsSubsequence("restore chat-1 chat-1.layers-4-8.kv",
				"restore bad bad.layers-4-8.kv", "evict chat-1");
	}

	@Test
	@DisplayName("tensor-parallel node: its rank in the label; no session directory: UNSUPPORTED, handler untouched")
	void tensorParallelRankAndNoDirectory() throws Exception {
		RecordingHandler handler = new RecordingHandler();
		var stub = start(handler, 0, 4, 1, 3);
		System.setProperty(SessionDirOptions.ENV, dir.toString());
		assertThat(stub.saveSessionKv(request("s")).getFile()).isEqualTo("s.layers-0-4-rank-1-of-3.kv");

		System.clearProperty(SessionDirOptions.ENV);
		org.junit.jupiter.api.Assumptions.assumeTrue(System.getenv(SessionDirOptions.ENV) == null,
				"the environment sets " + SessionDirOptions.ENV);
		SessionKvResponse none = stub.saveSessionKv(request("t"));
		assertThat(none.getErrorReason()).isEqualTo("UNSUPPORTED");
		assertThat(none.getError()).contains(SessionDirOptions.ENV);
		assertThat(handler.calls).hasSize(1);
	}

	private NodeServiceGrpc.NodeServiceBlockingStub start(RecordingHandler handler, int start, int end, int rank,
			int world) throws Exception {
		Path gguf = MetadataOnlyGguf.write(dir, "llama");
		int port = freePort();
		server = new EmbeddedNodeServer("n0", port, gguf.toString(), false, (model, ctx, backend, adapters) -> handler);
		server.start();
		channel = ManagedChannelBuilder.forAddress("localhost", port).usePlaintext().build();
		var stub = NodeServiceGrpc.newBlockingStub(channel).withDeadlineAfter(60, TimeUnit.SECONDS);
		assertThat(stub.loadShard(LoadShardRequest.newBuilder().setModelId("m").setStartLayer(start).setEndLayer(end)
				.setHasEmbeddings(start == 0).setHasOutputProjection(true).setTensorRank(rank)
				.setTensorWorldSize(world).build()).getSuccess()).isTrue();
		return stub;
	}

	private static SessionKvRequest request(String id) {
		return SessionKvRequest.newBuilder().setSessionId(id).setModel("m.gguf").setAdapter("a.lora")
				.addAllTokens(List.of(5, 6, 7)).build();
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
