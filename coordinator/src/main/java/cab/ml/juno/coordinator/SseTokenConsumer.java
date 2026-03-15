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

package cab.ml.juno.coordinator;

/**
 * TokenConsumer implementation that writes each token as an SSE JSON event.
 *
 * Decoupled from Javalin via the SseEmitter functional interface — in
 * production pass client::sendEvent; in tests pass a List::add.
 *
 * Event format (one per generated token): {"requestId":"...","token":"
 * world","tokenId":1917,"isComplete":false}
 *
 * Final event (sent by caller via sendComplete()):
 * {"requestId":"...","token":"","tokenId":0,"isComplete":true,"finishReason":"stop"}
 *
 * Must be non-blocking per TokenConsumer contract — JSON serialization is
 * synchronous but cheap. The SseEmitter write is the responsibility of the
 * caller to keep non-blocking (Javalin's SseClient buffers writes).
 */
public final class SseTokenConsumer implements TokenConsumer {

	/** Receives serialized JSON event strings, one per token. */
	@FunctionalInterface
	public interface SseEmitter {
		void emit(String data);
	}

	private final String requestId;
	private final SseEmitter emitter;

	public SseTokenConsumer(String requestId, SseEmitter emitter) {
		if (requestId == null || requestId.isBlank())
			throw new IllegalArgumentException("requestId must not be blank");
		if (emitter == null)
			throw new IllegalArgumentException("emitter must not be null");
		this.requestId = requestId;
		this.emitter = emitter;
	}

	@Override
	public void onToken(String piece, int tokenId, int position) {
		emitter.emit(buildEvent(piece, tokenId, false, null));
	}

	/**
	 * Send the terminal event with isComplete=true. Called by the route handler
	 * after the generation future completes.
	 *
	 * @param finishReason "stop", "length", or "error"
	 */
	public void sendComplete(String finishReason) {
		emitter.emit(buildEvent("", 0, true, finishReason));
	}

	// ── JSON serialization — no external deps ─────────────────────────────────

	private String buildEvent(String token, int tokenId, boolean isComplete, String finishReason) {
		StringBuilder sb = new StringBuilder(128);
		sb.append("{");
		appendString(sb, "requestId", requestId);
		sb.append(",");
		appendString(sb, "token", token);
		sb.append(",");
		appendInt(sb, "tokenId", tokenId);
		sb.append(",");
		appendBool(sb, "isComplete", isComplete);
		if (finishReason != null) {
			sb.append(",");
			appendString(sb, "finishReason", finishReason);
		}
		sb.append("}");
		return sb.toString();
	}

	private static void appendString(StringBuilder sb, String key, String value) {
		sb.append('"').append(key).append("\":\"").append(escape(value)).append('"');
	}

	private static void appendInt(StringBuilder sb, String key, int value) {
		sb.append('"').append(key).append("\":").append(value);
	}

	private static void appendBool(StringBuilder sb, String key, boolean value) {
		sb.append('"').append(key).append("\":").append(value);
	}

	/** Minimal JSON string escaping — handles the common cases. */
	private static String escape(String s) {
		if (s == null)
			return "";
		StringBuilder out = new StringBuilder(s.length());
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			switch (c) {
			case '"' -> out.append("\\\"");
			case '\\' -> out.append("\\\\");
			case '\n' -> out.append("\\n");
			case '\r' -> out.append("\\r");
			case '\t' -> out.append("\\t");
			default -> out.append(c);
			}
		}
		return out.toString();
	}
}