package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.api.grpc.LoadShardRequest;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

/**
 * Where a node's gRPC server listens. Bound to every interface it is reachable from other hosts (a deployed
 * cluster); bound to loopback (a local cluster's forked nodes) it is reachable only from this host, which is all
 * a local cluster needs, and Windows' firewall does not ask the user about it.
 */
@DisplayName("EmbeddedNodeServer — listen address")
class EmbeddedNodeServerBindTest {

	private EmbeddedNodeServer server;
	private ManagedChannel channel;

	@AfterEach
	void tearDown() throws Exception {
		if (channel != null)
			channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
		if (server != null)
			server.stop();
	}

	@Test
	@DisplayName("without a bind address the server listens on every interface")
	void without_a_bind_address_every_interface() throws Exception {
		server = new EmbeddedNodeServer("n0", freePort());
		server.start();
		List<InetSocketAddress> addresses = server.listenAddresses();
		assertThat(addresses).isNotEmpty();
		assertThat(addresses).allSatisfy(a -> assertThat(a.getAddress().isAnyLocalAddress()).isTrue());
	}

	@Test
	@DisplayName("bound to loopback, the server listens there only and answers a client at localhost")
	void bound_to_loopback_listens_there_only() throws Exception {
		int port = freePort();
		server = new EmbeddedNodeServer("n0", port, null, false, InetAddress.getLoopbackAddress());
		server.start();
		List<InetSocketAddress> addresses = server.listenAddresses();
		assertThat(addresses).isNotEmpty();
		assertThat(addresses).allSatisfy(a -> {
			assertThat(a.getAddress().isLoopbackAddress()).isTrue();
			assertThat(a.getPort()).isEqualTo(port);
		});

		channel = ManagedChannelBuilder.forAddress("localhost", port).usePlaintext().build();
		var reply = NodeServiceGrpc.newBlockingStub(channel).withDeadlineAfter(30, TimeUnit.SECONDS)
				.loadShard(LoadShardRequest.newBuilder().setModelId("m").setStartLayer(0).setEndLayer(2)
						.setHasEmbeddings(true).setHasOutputProjection(true).build());
		assertThat(reply.getSuccess()).isTrue();
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
