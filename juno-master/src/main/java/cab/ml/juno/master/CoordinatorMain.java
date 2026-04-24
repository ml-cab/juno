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
package cab.ml.juno.master;

import cab.ml.juno.apiserver.JunoApiServerMain;

/**
 * Standalone coordinator entry point for distributed AWS deployment.
 *
 * <p>Delegates to {@link JunoApiServerMain} so the shaded jar keeps this class as the manifest
 * {@code Main-Class} while HTTP/OpenAI implementation lives in the {@code juno-api-server} module.
 *
 * <p>Environment variables:
 * <pre>
 *   JUNO_NODE_ADDRESSES   comma-separated host:port list, one per node
 *                         e.g. "10.0.0.1:19092,10.0.0.2:19092,10.0.0.3:19092"
 *   JUNO_MODEL_PATH       local path to the GGUF file (for tokenizer + config)
 *   JUNO_PTYPE            "pipeline" (default) or "tensor"
 *   JUNO_HTTP_PORT        REST port (default: 8080)
 *   JUNO_DTYPE            FLOAT32 | FLOAT16 | INT8  (default: FLOAT16)
 *   JUNO_MAX_QUEUE        max scheduler queue depth (default: 1000)
 *   JUNO_USE_GPU          "true" / "false" propagated to nodes via LoadShard
 *
 *   Health sidecar (optional — starts alongside the coordinator, not instead of it):
 *   JUNO_HEALTH           "true" to enable the health HTTP sidecar (default: false)
 *   JUNO_HEALTH_PORT      port for the health sidecar (default: 8081)
 *   JUNO_HEALTH_STALE_MS  staleness threshold in ms (default: 15000)
 *   JUNO_HEALTH_WARN      VRAM warning threshold 0.0-1.0 (default: 0.90)
 *   JUNO_HEALTH_CRITICAL  VRAM critical threshold 0.0-1.0 (default: 0.98)
 * </pre>
 */
public final class CoordinatorMain {

	public static void main(String[] args) throws Exception {
		JunoApiServerMain.main(args);
	}
}
