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

import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.google.protobuf.ByteString;

import cab.ml.juno.api.grpc.ActivationDtype;
import cab.ml.juno.api.grpc.ForwardRequest;
import cab.ml.juno.api.grpc.ForwardResponse;
import cab.ml.juno.api.grpc.NodeServiceGrpc;
import cab.ml.juno.node.ActivationCodec;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * The cluster clients send a request's KV reservation ({@link cab.ml.juno.node.InferencePipeline#reserveKv}) to every
 * node with that request's next forward pass, once; a request evicted before its next pass sends none.
 */
@DisplayName("Pipeline and tensor-parallel clients carry the KV reservation to every node")
class PipelineClientKvReserveTest {

	private static final int VOCAB = 8;

	private final List<Server> servers = new ArrayList<>();

	@AfterEach
	void stop() {
		for (Server s : servers)
			s.shutdownNow();
	}

	/** A node that records every forward request; the last node answers logits, the others activations. */
	private List<ForwardRequest> node(int port, boolean last) throws Exception {
		List<ForwardRequest> seen = new CopyOnWriteArrayList<>();
		servers.add(ServerBuilder.forPort(port).addService(new NodeServiceGrpc.NodeServiceImplBase() {
			@Override
			public void forwardPass(ForwardRequest request, StreamObserver<ForwardResponse> out) {
				seen.add(request);
				byte[] payload = ActivationCodec.encode(new float[last ? VOCAB : 4],
						cab.ml.juno.node.ActivationDtype.FLOAT32);
				out.onNext(ForwardResponse.newBuilder().setRequestId(request.getRequestId())
						.setActivation(ByteString.copyFrom(payload)).setIsLastNode(last)
						.setDtype(ActivationDtype.FLOAT32).build());
				out.onCompleted();
			}
		}).build().start());
		return seen;
	}

	private static List<Integer> reserves(List<ForwardRequest> seen) {
		return seen.stream().map(ForwardRequest::getKvReservePositions).toList();
	}

	@Test
	@DisplayName("pipeline parallel: every stage gets the reservation with the next pass, and the pass after gets none")
	void pipelineStagesGetTheReservationOnce() throws Exception {
		int p0 = freePort(), p1 = freePort();
		List<ForwardRequest> first = node(p0, false);
		List<ForwardRequest> second = node(p1, true);
		ProcessPipelineClient client = new ProcessPipelineClient(List.of(
				new ProcessPipelineClient.NodeAddress("localhost", p0),
				new ProcessPipelineClient.NodeAddress("localhost", p1)), VOCAB);
		try {
			client.reserveKv("r1", 2112);
			client.forward("r1", new int[] { 1, 2, 3 }, 0);
			client.forward("r1", new int[] { 4 }, 3);
		} finally {
			client.shutdown();
		}
		assertThat(reserves(first)).containsExactly(2112, 0);
		assertThat(reserves(second)).containsExactly(2112, 0);
	}

	@Test
	@DisplayName("pipeline parallel: a request evicted before its next pass sends no reservation")
	void evictDropsAnUnsentReservation() throws Exception {
		int p0 = freePort();
		List<ForwardRequest> only = node(p0, true);
		ProcessPipelineClient client = new ProcessPipelineClient(
				List.of(new ProcessPipelineClient.NodeAddress("localhost", p0)), VOCAB);
		try {
			client.reserveKv("r1", 2112);
			client.evict("r1");
			client.forward("r1", new int[] { 1 }, 0);
		} finally {
			client.shutdown();
		}
		assertThat(reserves(only)).containsExactly(0);
	}

	@Test
	@DisplayName("tensor parallel: every rank gets the reservation with the next pass, and the pass after gets none")
	void tensorRanksGetTheReservationOnce() throws Exception {
		int p0 = freePort(), p1 = freePort();
		List<ForwardRequest> rank0 = node(p0, true);
		List<ForwardRequest> rank1 = node(p1, true);
		TensorParallelPipelineClient client = new TensorParallelPipelineClient(List.of(
				new TensorParallelPipelineClient.NodeAddress("localhost", p0),
				new TensorParallelPipelineClient.NodeAddress("localhost", p1)), VOCAB);
		try {
			client.reserveKv("r1", 2112);
			client.forward("r1", new int[] { 1, 2, 3 }, 0);
			client.forward("r1", new int[] { 4 }, 3);
		} finally {
			client.shutdown();
		}
		assertThat(reserves(rank0)).containsExactly(2112, 0);
		assertThat(reserves(rank1)).containsExactly(2112, 0);
	}

	private static int freePort() throws Exception {
		try (ServerSocket s = new ServerSocket(0)) {
			return s.getLocalPort();
		}
	}
}
