/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

import io.javalin.Javalin;
import io.javalin.http.Context;

/**
 * Minimal HTTP API for LoRA training in {@code ./juno lora} mode.
 *
 * <p>
 * Started when {@code --api-port} is set. Endpoints:
 * <ul>
 * <li>{@code POST /v1/lora/train-file-qa} — JSON array of {@code {"Q","A"}} (same
 * schema as the REPL file)</li>
 * <li>{@code POST /v1/lora/save} — persist the adapter checkpoint</li>
 * </ul>
 *
 * @author Yevhen Soldatov
 */
public final class LoraApiServer {

	private static final Logger log = Logger.getLogger(LoraApiServer.class.getName());

	/**
	 * Training session backed by the live LoRA REPL adapters.
	 */
	public interface Backend {
		LoraTrainingLoop.TrainingResult trainFileQa(List<LoraQaFile.Pair> pairs) throws Exception;

		Path save() throws Exception;
	}

	private final Backend backend;
	private Javalin app;

	public LoraApiServer(Backend backend) {
		this.backend = Objects.requireNonNull(backend, "backend");
	}

	public void start(int port) {
		if (port <= 0)
			throw new IllegalArgumentException("port must be > 0");
		app = Javalin.create(config -> {
			config.useVirtualThreads = true;
			config.showJavalinBanner = false;
		});
		app.post("/v1/lora/train-file-qa", this::handleTrainFileQa);
		app.post("/v1/lora/save", this::handleSave);
		app.exception(IllegalArgumentException.class, (e, ctx) -> {
			ctx.status(400).json(errorBody(400, "BAD_REQUEST", e.getMessage()));
		});
		app.exception(Exception.class, (e, ctx) -> {
			log.warning("LoRA API error: " + e);
			ctx.status(500).json(errorBody(500, "INTERNAL", e.getMessage() != null ? e.getMessage() : "error"));
		});
		app.start(port);
	}

	public void stop() {
		if (app != null) {
			app.stop();
			app = null;
		}
	}

	private void handleTrainFileQa(Context ctx) throws Exception {
		List<LoraQaFile.Pair> pairs = LoraQaFile.parse(ctx.body());
		LoraTrainingLoop.TrainingResult result = backend.trainFileQa(pairs);
		ctx.status(200).json(trainResponse(pairs.size(), result));
	}

	private void handleSave(Context ctx) throws Exception {
		Path path = backend.save();
		ctx.status(200).json(Map.of("saved", true, "path", path.toString()));
	}

	/** Package-visible for tests. */
	static Map<String, Object> trainResponse(int pairCount, LoraTrainingLoop.TrainingResult result) {
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("pairCount", pairCount);
		out.put("unitCount", pairCount * 4);
		out.put("finalTrainLoss", result.finalTrainLoss());
		out.put("passCount", result.passCount());
		out.put("optimizerUpdateCount", result.optimizerUpdateCount());
		out.put("stopReason", result.stopReason().name());
		out.put("targetReached", result.targetReached());
		return out;
	}

	private static Map<String, Object> errorBody(int code, String error, String message) {
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("code", code);
		out.put("error", error);
		out.put("message", message != null ? message : "");
		return out;
	}
}
