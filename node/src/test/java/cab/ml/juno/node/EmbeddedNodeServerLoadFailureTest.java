package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.ServerSocket;
import java.nio.file.Path;
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
import cab.ml.juno.api.grpc.LoadShardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

/**
 * A node started with a real model must never answer with stub output. If the
 * model cannot be loaded, the node reports the failure to the coordinator and
 * refuses forward passes, because a stub that returns fixed logits looks like a
 * healthy shard to every caller that does not read the message.
 */
@DisplayName("EmbeddedNodeServer — model load failure")
class EmbeddedNodeServerLoadFailureTest {

	@TempDir
	Path dir;

	private EmbeddedNodeServer server;
	private ManagedChannel channel;
	private NodeServiceGrpc.NodeServiceBlockingStub stub;

	@AfterEach
	void tearDown() throws Exception {
		if (channel != null)
			channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
		if (server != null)
			server.stop();
	}

	private void start(EmbeddedNodeServer node, int port) throws Exception {
		server = node;
		server.start();
		channel = ManagedChannelBuilder.forAddress("localhost", port).usePlaintext().build();
		stub = NodeServiceGrpc.newBlockingStub(channel).withDeadlineAfter(60, TimeUnit.SECONDS);
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}

	private static LoadShardRequest loadRequest() {
		return LoadShardRequest.newBuilder().setModelId("m").setStartLayer(0).setEndLayer(4).setHasEmbeddings(true)
				.setHasOutputProjection(true).build();
	}

	private static ForwardRequest forwardRequest() {
		byte[] tokens = java.nio.ByteBuffer.allocate(4).putInt(1).array();
		return ForwardRequest.newBuilder().setRequestId("r1").setModelId("m").setSequencePos(0)
				.setActivation(ByteString.copyFrom(tokens)).setDtype(ActivationDtype.FLOAT32).build();
	}

	@Test
	@DisplayName("an unsupported architecture is reported as a failed load, and the node then refuses forward passes")
	void unsupported_architecture_fails_the_load_and_the_forward_pass() throws Exception {
		Path gguf = MetadataOnlyGguf.write(dir, "gemma4");
		int port = freePort();
		start(new EmbeddedNodeServer("n0", port, gguf.toString(), false), port);

		LoadShardResponse load = stub.loadShard(loadRequest());

		assertThat(load.getSuccess()).isFalse();
		assertThat(load.getMessage()).contains("Unsupported model architecture").contains("'gemma4'");

		ForwardResponse forward = stub.forwardPass(forwardRequest());
		assertThat(forward.getError()).as("a node whose shard failed to load must not return stub logits")
				.isNotEmpty();
		assertThat(forward.getActivation()).isEqualTo(ByteString.EMPTY);
	}

	@Test
	@DisplayName("a missing model file is reported as a failed load")
	void missing_model_file_fails_the_load() throws Exception {
		int port = freePort();
		start(new EmbeddedNodeServer("n0", port, dir.resolve("does-not-exist.gguf").toString(), false), port);

		LoadShardResponse load = stub.loadShard(loadRequest());

		assertThat(load.getSuccess()).isFalse();
		assertThat(load.getMessage()).isNotBlank();
	}

	@Test
	@DisplayName("a real-model node refuses forward passes before any shard has been loaded")
	void real_model_node_refuses_forward_before_load() throws Exception {
		Path gguf = MetadataOnlyGguf.write(dir, "llama");
		int port = freePort();
		start(new EmbeddedNodeServer("n0", port, gguf.toString(), false), port);

		ForwardResponse forward = stub.forwardPass(forwardRequest());

		assertThat(forward.getError()).isNotEmpty();
	}

	@Test
	@DisplayName("stub mode (no model path) is unchanged: the load succeeds and forward passes are served")
	void stub_mode_is_unchanged() throws Exception {
		int port = freePort();
		start(new EmbeddedNodeServer("n0", port), port);

		LoadShardResponse load = stub.loadShard(loadRequest());
		ForwardResponse forward = stub.forwardPass(forwardRequest());

		assertThat(load.getSuccess()).isTrue();
		assertThat(forward.getError()).isEmpty();
	}
}
