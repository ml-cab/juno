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
 *   vision  →  node        (VisionEncoder, VisionAwareForwardPassHandler use node APIs)
 *   vision  →  coordinator (VisionChatHandler uses RequestScheduler)
 *   juno-master  →  vision (CoordinatorMain calls LlavaHandlerFactory.build())
 *   node    →  (nothing in vision)
 * </pre>
 *
 * Usage (in CoordinatorMain, after GGUF architecture has been detected):
 * <pre>{@code
 * if (LlavaHandlerFactory.isVisionArchitecture(modelPath)) {
 *     LlavaHandlerFactory.Built built = LlavaHandlerFactory.build(modelPath, shardContext);
 *     apiServer.withVisionHandler(new VisionChatHandler(
 *             scheduler, registry, built.encoder(), built.visionHandler()));
 *     // built.textHandler() is already wrapped inside built.visionHandler()
 *     // — do NOT pass it separately to LocalInferencePipeline
 *     pipeline = new LocalInferencePipeline(built.visionHandler(), ...);
 * }
 * }</pre>
 */
public final class LlavaHandlerFactory {

    private static final Logger log = Logger.getLogger(LlavaHandlerFactory.class.getName());

    /** GGUF architecture strings that indicate a multimodal LLaVA model. */
    private static final java.util.Set<String> VISION_ARCHS = java.util.Set.of(
            "llava", "llava-1.5", "llava-qwen2");

    private LlavaHandlerFactory() {}

    /**
     * Return value of {@link #build}: the wrapped handler, the raw text handler
     * it delegates to, and the vision encoder needed by @see VisionChatHandler.
     */
    public record Built(
            VisionAwareForwardPassHandler visionHandler,
            ForwardPassHandler textHandler,
            VisionEncoder encoder,
            VisionConfig config,
            int imageTokenId
    ) {}

    /**
     * True when the GGUF file's {@code general.architecture} is a known vision
     * architecture. Call this before {@link #build} to decide whether vision
     * wiring is needed.
     */
    public static boolean isVisionArchitecture(Path modelPath) throws IOException {
        try (GgufReader r = GgufReader.open(modelPath)) {
            String arch = r.metaString("general.architecture");
            return arch != null && VISION_ARCHS.contains(arch.toLowerCase().strip());
        }
    }

    /**
     * Build the vision-capable forward pass handler for a LLaVA model.
     *
     * Internally:
     * <ol>
     *   <li>Delegates to {@link ForwardPassHandlerLoader#load} to get the LLaMA
     *       text handler (loads all text-layer weights for the given shard).
     *   <li>Opens the GGUF a second time to load CLIP vision encoder weights.
     *   <li>Wraps the text handler in {@link VisionAwareForwardPassHandler}.
     * </ol>
     *
     * The {@code imageTokenId} is read from the system property
     * {@code juno.vision.image_token_id} (default 32000 for LLaVA-1.5).
     * Override for Phi-3 Vision: {@code -Djuno.vision.image_token_id=32044}.
     *
     * @param modelPath path to the GGUF file
     * @param context   shard assignment for this node
     * @param backend   compute backend (CPU or GPU MatVec)
     * @return {@link Built} containing all wired components
     * @throws IOException if the file cannot be opened or a required tensor is missing
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
        VisionAwareForwardPassHandler visionHandler =
                new VisionAwareForwardPassHandler(textHandler, imageTokenId, vCfg.projectionDim());

        log.info("Vision handler ready — imageTokenId=" + imageTokenId
                + "  patches=" + vCfg.numPatches()
                + "  projDim=" + vCfg.projectionDim());

        return new Built(visionHandler, textHandler, encoder, vCfg, imageTokenId);
    }
}