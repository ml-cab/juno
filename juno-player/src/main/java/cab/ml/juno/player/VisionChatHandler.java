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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import cab.ml.juno.coordinator.GenerationResult;
import cab.ml.juno.coordinator.InferenceRequest;
import cab.ml.juno.coordinator.RequestPriority;
import cab.ml.juno.coordinator.RequestScheduler;
import cab.ml.juno.coordinator.TokenConsumer;
import cab.ml.juno.registry.ModelRegistry;
import cab.ml.juno.registry.ModelStatus;
import cab.ml.juno.sampler.SamplingParams;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.vision.ImagePatchEmbedder;
import cab.ml.juno.vision.VisionAwareForwardPassHandler;
import cab.ml.juno.vision.VisionEncoder;
import io.javalin.http.Context;

/**
 * Javalin handler for multimodal (image-to-text) inference.
 *
 * Routes:
 * <pre>
 *   POST /v1/vision/chat                  — blocking JSON response
 *   POST /v1/vision/chat/stream           — SSE streaming response
 * </pre>
 *
 * Request format: {@code multipart/form-data} with two parts:
 * <ul>
 *   <li>{@code image}  — binary image file (JPEG, PNG, GIF, BMP)
 *   <li>{@code request} — JSON body matching {@link VisionChatRequest}
 * </ul>
 *
 * The handler:
 * <ol>
 *   <li>Reads the {@code image} part and the {@code request} JSON part.
 *   <li>Validates the model is loaded and is vision-capable.
 *   <li>Encodes the image with {@link VisionEncoder} → patch embeddings.
 *   <li>Registers embeddings on the {@link VisionAwareForwardPassHandler}.
 *   <li>Builds an {@link InferenceRequest} with image-token placeholders in the
 *       text and submits it to the {@link RequestScheduler}.
 *   <li>On completion, releases the patch embeddings.
 *   <li>Returns an OpenAI-compatible JSON response body.
 * </ol>
 *
 * Image tokens are represented in the text as a run of {@code <image>}
 * placeholder strings that the tokenizer will encode to the model-specific
 * image token ID.  The exact number of placeholders equals
 * {@code VisionConfig.numPatches()}.
 *
 * Thread-safe: stateless except for the shared {@link VisionAwareForwardPassHandler}
 * and {@link VisionEncoder}, both of which are themselves thread-safe.
 */
public final class VisionChatHandler {

    private static final Logger log = Logger.getLogger(VisionChatHandler.class.getName());
    private static final ObjectMapper JSON = new ObjectMapper();

    private final RequestScheduler scheduler;
    private final ModelRegistry registry;
    private final VisionEncoder encoder;
    private final VisionAwareForwardPassHandler visionHandler;

    /**
     * @param scheduler    shared request scheduler
     * @param registry     model registry (used to validate model availability)
     * @param encoder      vision encoder (pre-loaded weights)
     * @param visionHandler handler that injects patch embeddings into the LLM forward pass
     */
    public VisionChatHandler(RequestScheduler scheduler,
                              ModelRegistry registry,
                              VisionEncoder encoder,
                              VisionAwareForwardPassHandler visionHandler) {
        this.scheduler      = scheduler;
        this.registry       = registry;
        this.encoder        = encoder;
        this.visionHandler  = visionHandler;
    }

    // ── Route handlers ────────────────────────────────────────────────────

    /** Handle POST /v1/vision/chat — blocking JSON. */
    public void handleBlocking(Context ctx) {
        VisionChatRequest req = parseRequest(ctx);
        if (req == null) return;

        byte[] imageBytes = readImagePart(ctx);
        if (imageBytes == null) return;

        String modelId = resolveModel(ctx, req.model());
        if (modelId == null) return;

        GenerationResult result = runInference(ctx, req, imageBytes, modelId, null);
        if (result == null) return;

        ctx.status(200).json(buildResponse(result, modelId));
    }

    /** Handle POST /v1/vision/chat/stream — SSE streaming. */
    public void handleStreaming(Context ctx) {
        VisionChatRequest req = parseRequest(ctx);
        if (req == null) return;

        byte[] imageBytes = readImagePart(ctx);
        if (imageBytes == null) return;

        String modelId = resolveModel(ctx, req.model());
        if (modelId == null) return;

        ctx.res().setContentType("text/event-stream");
        ctx.res().setCharacterEncoding("UTF-8");
        ctx.res().setHeader("Cache-Control", "no-cache");
        ctx.res().setHeader("X-Accel-Buffering", "no");

        StringBuilder buf = new StringBuilder();
        TokenConsumer consumer = (piece, tokenId, pos) -> {
            buf.append(piece != null ? piece : "");
            writeChunk(ctx, piece != null ? piece : "");
        };

        runInference(ctx, req, imageBytes, modelId, consumer);
        writeChunk(ctx, "[DONE]");
    }

    // ── Core inference ────────────────────────────────────────────────────

    /**
     * Encode the image, register embeddings, submit request, release embeddings.
     *
     * @param consumer null for blocking mode
     * @return GenerationResult, or null if an error was already written to ctx
     */
    private GenerationResult runInference(Context ctx, VisionChatRequest req,
                                           byte[] imageBytes, String modelId,
                                           TokenConsumer consumer) {
        // Encode the image into patch embeddings
        float[][] patches;
        try {
            ImagePatchEmbedder embedder = new ImagePatchEmbedder(encoder.config());
            float[] pixelTensor = embedder.toPixelTensor(imageBytes);
            patches = encoder.encode(pixelTensor);
        } catch (IOException e) {
            log.warning("Image decoding failed: " + e.getMessage());
            error(ctx, 400, "invalid_image", "Could not decode the supplied image: " + e.getMessage());
            return null;
        }

        // Build the text prompt with image token placeholders
        List<ChatMessage> messages = buildMessagesWithImageTokens(req, patches.length);

        // Build InferenceRequest
        SamplingParams sampling = buildSampling(req);
        RequestPriority priority = RequestPriority.NORMAL;
        InferenceRequest inferenceReq = InferenceRequest.of(modelId, messages, sampling, priority);

        // Register patch embeddings so VisionAwareForwardPassHandler can inject them
        visionHandler.registerVisionEmbeddings(inferenceReq.requestId(), patches);

        try {
            if (consumer != null) {
                return scheduler.submit(inferenceReq, consumer).join();
            } else {
                return scheduler.submitAndWait(inferenceReq);
            }
        } catch (RequestScheduler.QueueFullException e) {
            error(ctx, 429, "rate_limit_exceeded", e.getMessage());
            return null;
        } catch (Exception e) {
            log.warning("Vision inference error: " + e.getMessage());
            error(ctx, 500, "internal_error", "Inference failed: " + e.getMessage());
            return null;
        } finally {
            visionHandler.releaseVisionEmbeddings(inferenceReq.requestId());
        }
    }

    // ── Message construction ──────────────────────────────────────────────

    /**
     * Build the message list with image placeholder tokens injected before the
     * user text.  The image is represented as {@code numPatches} repetitions of
     * the {@code <image>} token string, which the model's tokenizer maps to the
     * special image token ID.
     */
    private List<ChatMessage> buildMessagesWithImageTokens(VisionChatRequest req, int numPatches) {
        List<ChatMessage> out = new ArrayList<>();

        // Carry over any system message from the request
        if (req.messages() != null) {
            for (VisionChatRequest.VisionMessage m : req.messages()) {
                if ("system".equals(m.role())) {
                    out.add(ChatMessage.system(m.content()));
                }
            }
        }

        // Build image placeholder string: one <image> token per patch
        String imagePlaceholder = "<image>".repeat(numPatches);

        // Prepend image placeholder to user message
        String userText = "";
        if (req.messages() != null) {
            for (VisionChatRequest.VisionMessage m : req.messages()) {
                if ("user".equals(m.role())) {
                    userText = m.content() != null ? m.content() : "";
                    break;
                }
            }
        }
        out.add(ChatMessage.user(imagePlaceholder + "\n" + userText));
        return out;
    }

    // ── Request parsing ───────────────────────────────────────────────────

    private VisionChatRequest parseRequest(Context ctx) {
        String json = ctx.formParam("request");
        if (json == null || json.isBlank()) {
            error(ctx, 400, "invalid_request", "Missing 'request' form field (JSON)");
            return null;
        }
        try {
            return JSON.readValue(json, VisionChatRequest.class);
        } catch (Exception e) {
            error(ctx, 400, "invalid_request", "Could not parse 'request' JSON: " + e.getMessage());
            return null;
        }
    }

    private byte[] readImagePart(Context ctx) {
        var part = ctx.uploadedFile("image");
        if (part == null) {
            error(ctx, 400, "invalid_request", "Missing 'image' form file part");
            return null;
        }
        try {
            return part.content().readAllBytes();
        } catch (IOException e) {
            error(ctx, 400, "invalid_request", "Could not read image bytes: " + e.getMessage());
            return null;
        }
    }

    private String resolveModel(Context ctx, String requested) {
        if (requested != null && !requested.isBlank()) {
            if (!registry.isLoaded(requested)) {
                error(ctx, 503, "service_unavailable", "Model '" + requested + "' is not loaded");
                return null;
            }
            return requested;
        }
        // Fall back to the first model in LOADED status
        return registry.listModels().stream()
                .filter(m -> m.status() == ModelStatus.LOADED)
                .map(m -> m.modelId())
                .findFirst()
                .orElseGet(() -> {
                    error(ctx, 503, "service_unavailable", "No model is currently loaded");
                    return null;
                });
    }

    private SamplingParams buildSampling(VisionChatRequest req) {
        SamplingParams p = SamplingParams.defaults();
        if (req.maxTokens() != null)    p = p.withMaxTokens(req.maxTokens());
        if (req.temperature() != null)  p = p.withTemperature(req.temperature().floatValue());
        if (req.topP() != null)         p = p.withTopP(req.topP().floatValue());
        return p;
    }

    // ── Response building ─────────────────────────────────────────────────

    private static Map<String, Object> buildResponse(GenerationResult result, String modelId) {
        Map<String, Object> choice = Map.of(
                "index", 0,
                "message", Map.of("role", "assistant", "content", result.text()),
                "finish_reason", finishReason(result));
        Map<String, Object> usage = Map.of(
                "prompt_tokens",     result.promptTokens(),
                "completion_tokens", result.generatedTokens(),
                "total_tokens",      result.promptTokens() + result.generatedTokens());
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("id",      "vizcmpl-" + result.requestId().replace("-", "").substring(0, 16));
        root.put("object",  "vision.completion");
        root.put("model",   modelId);
        root.put("choices", List.of(choice));
        root.put("usage",   usage);
        root.put("x_juno_latency_ms", result.latency().toMillis());
        return root;
    }

    private static String finishReason(GenerationResult r) {
        return switch (r.stopReason()) {
            case EOS_TOKEN, STOP_TOKEN -> "stop";
            case MAX_TOKENS            -> "length";
            case ERROR                 -> "error";
        };
    }

    private static void writeChunk(Context ctx, String text) {
        try {
            ctx.res().getWriter().write("data: " + text + "\n\n");
            ctx.res().getWriter().flush();
        } catch (IOException e) {
            log.fine("SSE write failed: " + e.getMessage());
        }
    }

    private static void error(Context ctx, int status, String code, String message) {
        ctx.status(status).json(Map.of("error", Map.of("code", code, "message", message)));
    }

    // ── Request / message DTOs ────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record VisionChatRequest(
            @JsonProperty("model")       String model,
            @JsonProperty("messages")    List<VisionMessage> messages,
            @JsonProperty("max_tokens")  Integer maxTokens,
            @JsonProperty("temperature") Double temperature,
            @JsonProperty("top_p")       Double topP
    ) {
        @JsonIgnoreProperties(ignoreUnknown = true)
        public record VisionMessage(
                @JsonProperty("role")    String role,
                @JsonProperty("content") String content
        ) {}
    }
}