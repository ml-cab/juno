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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.api.grpc.SessionKvRequest;
import cab.ml.juno.api.grpc.SessionKvResponse;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionFileException.Reason;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.node.SessionKvPart;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * The cluster clients save, restore and drop a session's KV on every node ({@link InferencePipeline#saveKv},
 * {@link InferencePipeline#restoreKv}, {@link InferencePipeline#dropKv}), each node under its own directory; a node's
 * refusal comes back as its reason, and a restore refused anywhere drops the session on every node.
 */
@DisplayName("Pipeline and tensor-parallel clients carry session save, restore and drop to every node")
class PipelineClientSessionKvTest {

	private final List<Server> servers = new ArrayList<>();
	private final KvSessionStamp stamp = new KvSessionStamp("m.gguf", "a.lora", "all", new int[] { 4, 5, 6 });

	@AfterEach
	void stop() {
		for (Server s : servers)
			s.shutdownNow();
	}

	/** A node labelled {@code layers-i}; it refuses a restore of "bad" when {@code refuse} is set, and fails "io". */
	private List<String> node(int port, int i, boolean refuse) throws Exception {
		List<String> seen = new CopyOnWriteArrayList<>();
		servers.add(ServerBuilder.forPort(port).addService(new NodeServiceGrpc.NodeServiceImplBase() {
			@Override
			public void saveSessionKv(SessionKvRequest r, StreamObserver<SessionKvResponse> out) {
				seen.add("save " + r.getSessionId() + " " + r.getModel() + " " + r.getAdapter() + " " + r.getTokensList());
				SessionKvResponse.Builder b = answer(r, i);
				if (r.getSessionId().equals("io"))
					b.setErrorReason("IO").setError("disk full");
				out.onNext(b.build());
				out.onCompleted();
			}

			@Override
			public void restoreSessionKv(SessionKvRequest r, StreamObserver<SessionKvResponse> out) {
				seen.add("restore " + r.getSessionId());
				SessionKvResponse.Builder b = answer(r, i);
				if (refuse && r.getSessionId().equals("bad"))
					b.setErrorReason("MISMATCH").setError("node " + i + ": model differs");
				out.onNext(b.build());
				out.onCompleted();
			}

			@Override
			public void dropSessionKv(SessionKvRequest r, StreamObserver<SessionKvResponse> out) {
				seen.add("drop " + r.getSessionId());
				out.onNext(SessionKvResponse.newBuilder().setStage("layers-" + i).build());
				out.onCompleted();
			}
		}).build().start());
		return seen;
	}

	private static SessionKvResponse.Builder answer(SessionKvRequest r, int i) {
		return SessionKvResponse.newBuilder().setStage("layers-" + i).setFile(r.getSessionId() + ".layers-" + i + ".kv")
				.setBytes(10L * (i + 1));
	}

	@Test
	@DisplayName("pipeline parallel: parts from every node in order; a refusal drops the session everywhere")
	void pipelineParallel() throws Exception {
		int p0 = freePort(), p1 = freePort();
		List<String> n0 = node(p0, 0, false);
		List<String> n1 = node(p1, 1, true);
		ProcessPipelineClient client = new ProcessPipelineClient(List.of(
				new ProcessPipelineClient.NodeAddress("localhost", p0),
				new ProcessPipelineClient.NodeAddress("localhost", p1)), 8);
		try {
			exercise(client, n0, n1);
		} finally {
			client.shutdown();
		}
	}

	@Test
	@DisplayName("tensor parallel: the same, on every node")
	void tensorParallel() throws Exception {
		int p0 = freePort(), p1 = freePort();
		List<String> n0 = node(p0, 0, false);
		List<String> n1 = node(p1, 1, true);
		TensorParallelPipelineClient client = new TensorParallelPipelineClient(List.of(
				new TensorParallelPipelineClient.NodeAddress("localhost", p0),
				new TensorParallelPipelineClient.NodeAddress("localhost", p1)), 8);
		try {
			exercise(client, n0, n1);
		} finally {
			client.shutdown();
		}
	}

	private void exercise(InferencePipeline client, List<String> n0, List<String> n1) throws Exception {
		Path unused = Path.of("unused");
		assertThat(client.saveKv("s", stamp, unused)).containsExactly(new SessionKvPart("layers-0", "s.layers-0.kv", 10),
				new SessionKvPart("layers-1", "s.layers-1.kv", 20));
		assertThat(n0).containsExactly("save s m.gguf a.lora [4, 5, 6]");
		assertThat(n1).containsExactly("save s m.gguf a.lora [4, 5, 6]");

		assertThat(client.restoreKv("s", stamp, unused)).hasSize(2);
		assertThatThrownBy(() -> client.restoreKv("bad", stamp, unused))
				.isInstanceOfSatisfying(KvSessionFileException.class, e -> {
					assertThat(e.reason()).isEqualTo(Reason.MISMATCH);
					assertThat(e.getMessage()).contains("model differs");
				});
		assertThat(n0).endsWith("restore bad", "drop bad");
		assertThat(n1).endsWith("restore bad", "drop bad");

		assertThatThrownBy(() -> client.saveKv("io", stamp, unused)).isInstanceOf(IOException.class)
				.hasMessageContaining("disk full");

		client.dropKv("s");
		assertThat(n0).last().isEqualTo("drop s");
		assertThat(n1).last().isEqualTo("drop s");
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
