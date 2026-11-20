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
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

import cab.ml.juno.api.grpc.NodeServiceGrpc.NodeServiceBlockingStub;
import cab.ml.juno.api.grpc.SessionKvRequest;
import cab.ml.juno.api.grpc.SessionKvResponse;
import cab.ml.juno.kvcache.KvSessionFileException;
import cab.ml.juno.kvcache.KvSessionStamp;
import cab.ml.juno.node.SessionKvPart;

/**
 * Session save, restore and drop over every node of a cluster, for both cluster clients: each node writes or reads its
 * own layers under its own session directory, so a saved session is pinned to the nodes that saved it unless their
 * directories are shared. A node's refusal comes back as a {@link KvSessionFileException} of its reason.
 */
final class NodeSessionKv {

	private NodeSessionKv() {
	}

	static List<SessionKvPart> save(List<NodeServiceBlockingStub> nodes, String sessionId, KvSessionStamp stamp)
			throws IOException {
		return each(nodes, sessionId, stamp, NodeServiceBlockingStub::saveSessionKv);
	}

	/** As {@link #save}; if any node refuses, the session is dropped on every node before the refusal is passed on. */
	static List<SessionKvPart> restore(List<NodeServiceBlockingStub> nodes, String sessionId, KvSessionStamp stamp)
			throws IOException {
		try {
			return each(nodes, sessionId, stamp, NodeServiceBlockingStub::restoreSessionKv);
		} catch (IOException | RuntimeException e) {
			try {
				drop(nodes, sessionId);
			} catch (RuntimeException suppressed) {
				e.addSuppressed(suppressed);
			}
			throw e;
		}
	}

	static void drop(List<NodeServiceBlockingStub> nodes, String sessionId) {
		SessionKvRequest request = SessionKvRequest.newBuilder().setSessionId(sessionId).build();
		for (NodeServiceBlockingStub node : nodes)
			node.dropSessionKv(request);
	}

	private static List<SessionKvPart> each(List<NodeServiceBlockingStub> nodes, String sessionId,
			KvSessionStamp stamp, BiFunction<NodeServiceBlockingStub, SessionKvRequest, SessionKvResponse> call)
			throws IOException {
		SessionKvRequest.Builder b = SessionKvRequest.newBuilder().setSessionId(sessionId).setModel(stamp.model())
				.setAdapter(stamp.adapter());
		for (int t : stamp.tokens())
			b.addTokens(t);
		SessionKvRequest request = b.build();
		List<SessionKvPart> parts = new ArrayList<>(nodes.size());
		for (NodeServiceBlockingStub node : nodes)
			parts.add(part(call.apply(node, request)));
		return parts;
	}

	private static SessionKvPart part(SessionKvResponse r) throws IOException {
		String reason = r.getErrorReason();
		if (reason.isEmpty())
			return new SessionKvPart(r.getStage(), r.getFile(), r.getBytes());
		if (reason.equals("IO"))
			throw new IOException(r.getError());
		for (KvSessionFileException.Reason known : KvSessionFileException.Reason.values())
			if (known.name().equals(reason))
				throw new KvSessionFileException(known, r.getError());
		throw new IllegalStateException(r.getError());
	}
}
