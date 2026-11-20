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

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.KvSessionFileException;
import io.javalin.Javalin;
import io.javalin.http.Context;

/**
 * The session routes over a {@link SessionStore}. Each answers {@code session_id}, {@code tokens}, {@code bytes} and
 * {@code elapsed_ms}; a refusal answers 400 (no session directory, a bad id, a file of other KV or a damaged one, a
 * session that may not be saved), 404 (nothing saved under the id, or a session with no KV to save) or 409 (a request
 * of the session in flight).
 */
final class SessionRoutes {

	private static final Logger log = Logger.getLogger(SessionRoutes.class.getName());

	private final SessionStore store;

	SessionRoutes(SessionStore store) {
		this.store = store;
	}

	void register(Javalin app) {
		app.post("/v1/sessions/{session_id}/save", ctx -> answer(ctx, () -> store.save(id(ctx))));
		app.post("/v1/sessions/{session_id}/restore", ctx -> answer(ctx, () -> store.restore(id(ctx))));
		app.delete("/v1/sessions/{session_id}", ctx -> answer(ctx, () -> store.delete(id(ctx))));
	}

	private static String id(Context ctx) {
		return ctx.pathParam("session_id");
	}

	private static void answer(Context ctx, RequestScheduler.SessionCall<SessionStore.Outcome> call) {
		try {
			SessionStore.Outcome o = call.run();
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("session_id", o.sessionId());
			body.put("tokens", o.tokens());
			body.put("bytes", o.bytes());
			body.put("elapsed_ms", o.elapsedMs());
			ctx.status(200).json(body);
		} catch (SessionBusyException e) {
			error(ctx, 409, "CONFLICT", e.getMessage());
		} catch (KvSessionFileException e) {
			if (e.reason() == KvSessionFileException.Reason.NOT_FOUND)
				error(ctx, 404, "NOT_FOUND", e.getMessage());
			else
				error(ctx, 400, "BAD_REQUEST", e.getMessage());
		} catch (IllegalArgumentException e) {
			error(ctx, 400, "BAD_REQUEST", e.getMessage());
		} catch (IOException | RuntimeException e) {
			log.log(Level.WARNING, "Session call failed: " + ctx.method() + " " + ctx.path(), e);
			error(ctx, 500, "INTERNAL_ERROR", e.getMessage() != null ? e.getMessage() : e.getClass().getName());
		}
	}

	private static void error(Context ctx, int status, String code, String message) {
		ctx.status(status).json(Map.of("code", status, "error", code, "message", message));
	}
}
