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

package cab.ml.juno.vision;

import java.io.IOException;
import java.nio.file.Path;
import java.util.logging.Logger;

import cab.ml.juno.node.ForwardPassHandler;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.MatVec;
import cab.ml.juno.node.ShardContext;

/**
 * Factory for vision-capable forward pass handlers.
 *
 * Lives in the {@code vision} module — not in {@code node} — to avoid a
 * dependency cycle. The {@code node} module knows nothing about {@code vision};
 * the wiring is done here, above both.
 *
 * Dependency graph (no cycle):
 * <pre>
 *   vision      →  node, registry, tokenizer   (no coordinator)
 *   coordinator →  node, registry, ...          (no vision)
 *   juno-player →  vision + coordinator + node  (only module touching both)
 *   node        →  (nothing in vision)
 * </pre>
 *
 * Usage (in ConsoleMain.prepareVisionHandler(), BEFORE LocalInferencePipeline
 * is built — see that method's javadoc for why the ordering matters):
 * <pre>{@code
 * Path mmproj = mmprojPath != null ? Path.of(mmprojPath) : null;
 * LlavaHandlerFactory.Built built = null;
 * if (LlavaHandlerFactory.isVisionArchitecture(Path.of(modelPath), mmproj)) {
 *     // buildFromHandlers() replaces handlers.get(0) in place with a
 *     // vision-aware wrapper. This MUST happen before
 *     // LocalInferencePipeline.from(shardMap, handlers, ...) reads the list,
 *     // since that call snapshots each handler reference at construction time
 *     // and never re-reads the list afterwards.
 *     built = LlavaHandlerFactory.buildFromHandlers(
 *             Path.of(modelPath), mmproj, handlers, config);
 * }
 * var pipeline = LocalInferencePipeline.from(shardMap, handlers, ...); // sees the wrap
 * ...
 * // Later, once apiServer exists, register the routes:
 * if (built != null) {
 *     VisionChatHandler handler = new VisionChatHandler(
 *             scheduler, registry, built.encoder(), built.visionHandler());
 *     apiServer.addRoutes(app -> {
 *         app.post("/v1/vision/chat",        handler::handleBlocking);
 *         app.post("/v1/vision/chat/stream", handler::handleStreaming);
 *     });
 * }
 * }</pre>
 */
public final class LlavaHandlerFactory {

    private static final Logger log = Logger.getLogger(LlavaHandlerFactory.class.getName());

    private LlavaHandlerFactory() {}

    /**
     * Return value of {@link #build}: all wired components needed by VisionChatHandler.
     */
    public record Built(
            VisionAwareForwardPassHandler visionHandler,
            ForwardPassHandler textHandler,
            VisionEncoder encoder,
            VisionConfig config,
            int imageTokenId
    ) {}

    /**
     * True when the resolved vision-weights file (see {@link VisionModelPaths})
     * contains a CLIP vision encoder, indicating a multimodal LLaVA-family
     * model.
     *
     * Detection is based on the presence of the vision patch embedding tensor
     * {@code v.patch_embd.weight}, NOT on {@code general.architecture}.
     * LLaVA-1.5 (and variants) store {@code general.architecture = "llama"}
     * because the backbone is LLaMA — checking the architecture string always
     * returns false for these models.
     *
     * <p>Real GGUF releases of multimodal models split the CLIP encoder into
     * a separate {@code mmproj-*.gguf} file (see {@link VisionModelPaths}).
     * Pass that file as {@code mmprojPath} — checking the base model file
     * alone will always return {@code false} for these models.
     *
     * @param modelPath  path to the base LLM GGUF
     * @param mmprojPath path to a separate mmproj GGUF, or {@code null} if the
     *                   caller believes vision tensors are merged into
     *                   {@code modelPath} itself
     */
    public static boolean isVisionArchitecture(Path modelPath, Path mmprojPath) throws IOException {
        VisionModelPaths paths = VisionModelPaths.of(modelPath, mmprojPath);
        try (GgufReader r = GgufReader.open(paths.visionWeightsPath())) {
            String arch = r.metaString("general.architecture");
            boolean hasPatchEmbd = r.hasTensor("v.patch_embd.weight");
            log.info("[vision] isVisionArchitecture check — general.architecture=\"" + arch
                    + "\"  hasTensor(v.patch_embd.weight)=" + hasPatchEmbd
                    + "  visionWeightsPath=" + paths.visionWeightsPath()
                    + "  usesSeparateMmproj=" + paths.usesSeparateMmproj());
            return hasPatchEmbd;
        }
    }

    /**
     * Backward-compatible overload: probes {@code modelPath} itself for vision
     * tensors (merged-file assumption). Prefer
     * {@link #isVisionArchitecture(Path, Path)} — real GGUF releases keep the
     * CLIP encoder in a separate mmproj file, which this overload cannot see.
     */
    public static boolean isVisionArchitecture(Path modelPath) throws IOException {
        return isVisionArchitecture(modelPath, null);
    }

    /**
     * Build a vision-capable handler by wrapping already-loaded text handlers.
     *
     * This is the correct entry point for juno-player, where
     * {@code ForwardPassHandlerLoader} has already loaded the LLaMA text layers
     * into the pipeline. Wrapping those handlers avoids loading the multi-GB
     * GGUF a second time and ensures the {@link VisionAwareForwardPassHandler}
     * that receives inference requests is the same instance the
     * {@link cab.ml.juno.node.LocalInferencePipeline} uses.
     *
     * Only the CLIP vision encoder weights are read from disk — from
     * {@code mmprojPath} when given (the real-world case for every known
     * public LLaVA/Qwen-VL/SmolVLM GGUF release), or from {@code modelPath}
     * otherwise. Either way this is a small fraction of the file: only the
     * {@code v.*} and {@code mm.*} tensors are loaded.
     *
     * @param modelPath  path to the base LLM GGUF (for CLIP encoder weights
     *                   only when {@code mmprojPath} is {@code null})
     * @param mmprojPath path to a separate mmproj GGUF holding the CLIP vision
     *                   encoder, or {@code null} to probe {@code modelPath}
     *                   itself (merged-file fallback)
     * @param handlers   the already-loaded {@link ForwardPassHandler} list from
     *                   {@code runLocalRepl} — the first handler is wrapped
     * @param config     parsed LlamaConfig (for projectionDim cross-check)
     */
    public static Built buildFromHandlers(Path modelPath, Path mmprojPath,
                                          java.util.List<ForwardPassHandler> handlers,
                                          cab.ml.juno.node.LlamaConfig config) throws IOException {
        VisionModelPaths paths = VisionModelPaths.of(modelPath, mmprojPath);
        log.info("[vision] buildFromHandlers — visionWeightsPath=" + paths.visionWeightsPath()
                + "  usesSeparateMmproj=" + paths.usesSeparateMmproj()
                + "  handlers=" + (handlers == null ? "null" : handlers.size()));
        if (handlers == null || handlers.isEmpty())
            throw new IllegalArgumentException("handlers must not be empty");

        ForwardPassHandler textHandler = handlers.get(0);
        log.info("[vision] textHandler type=" + textHandler.getClass().getName());

        VisionConfig vCfg;
        VisionEncoder encoder;
        try (GgufReader r = GgufReader.open(paths.visionWeightsPath())) {
            vCfg    = VisionConfig.from(r);
            log.info("[vision] VisionConfig=" + vCfg);
            encoder = VisionEncoder.load(r, vCfg, cab.ml.juno.node.CpuMatVec.INSTANCE);
            log.info("[vision] VisionEncoder loaded");
        }

        int imageTokenId = Integer.getInteger("juno.vision.image_token_id", 32000);
        // Use encoder.outputDim() — derived from mm.0.weight's own GGUF shape —
        // not vCfg.projectionDim() (clip.vision.projection_dim metadata), which
        // is not reliable across mmproj exports; see VisionEncoder.outputDim().
        VisionAwareForwardPassHandler visionHandler =
                new VisionAwareForwardPassHandler(textHandler, imageTokenId, encoder.outputDim());

        // Replace handlers[0] with the vision-aware wrapper so the pipeline uses it
        handlers.set(0, visionHandler);
        log.info("[vision] handlers[0] replaced with VisionAwareForwardPassHandler"
                + "  imageTokenId=" + imageTokenId
                + "  patches=" + vCfg.numPatches()
                + "  outputDim=" + encoder.outputDim());

        return new Built(visionHandler, textHandler, encoder, vCfg, imageTokenId);
    }

    /**
     * Backward-compatible overload: reads vision weights from {@code modelPath}
     * itself (merged-file assumption). Prefer
     * {@link #buildFromHandlers(Path, Path, java.util.List, cab.ml.juno.node.LlamaConfig)}
     * with an explicit mmproj path — real GGUF releases keep the CLIP encoder
     * in a separate file.
     */
    public static Built buildFromHandlers(Path modelPath,
                                          java.util.List<ForwardPassHandler> handlers,
                                          cab.ml.juno.node.LlamaConfig config) throws IOException {
        return buildFromHandlers(modelPath, null, handlers, config);
    }

    /**
     * Build by loading a fresh set of text handlers from disk.
     * Use {@link #buildFromHandlers} in juno-player where handlers are already loaded.
     */
    public static Built build(Path modelPath, ShardContext context, MatVec backend) throws IOException {
        log.info("Building vision-aware handler for LLaVA model: " + modelPath);

        // Step 1: load the text (LLaMA backbone) layers via the standard loader.
        // ForwardPassHandlerLoader is in the node module — no vision import needed there.
        ForwardPassHandler textHandler = ForwardPassHandlerLoader.load(modelPath, context, backend);

        // Step 2: load CLIP vision encoder weights from the same GGUF.
        VisionConfig vCfg;
        VisionEncoder encoder;
        try (GgufReader r = GgufReader.open(modelPath)) {
            vCfg    = VisionConfig.from(r);
            encoder = VisionEncoder.load(r, vCfg, backend);
        }

        // Step 3: wrap the text handler with the vision embedding injector.
        int imageTokenId = Integer.getInteger("juno.vision.image_token_id", 32000);
        // encoder.outputDim() (from mm.0.weight's own shape), not
        // vCfg.projectionDim() (unreliable metadata) — see VisionEncoder.outputDim().
        VisionAwareForwardPassHandler visionHandler =
                new VisionAwareForwardPassHandler(textHandler, imageTokenId, encoder.outputDim());

        log.info("Vision handler ready — imageTokenId=" + imageTokenId
                + "  patches=" + vCfg.numPatches()
                + "  outputDim=" + encoder.outputDim());

        return new Built(visionHandler, textHandler, encoder, vCfg, imageTokenId);
    }
}