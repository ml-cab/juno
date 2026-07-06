package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Instant;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ModelIdResolver — resolves a client-supplied model id against ModelRegistry")
class ModelIdResolverTest {

    private ModelRegistry registry;

    @BeforeEach
    void setUp() {
        registry = new ModelRegistry(ShardPlanner.create());
    }

    private static ModelDescriptor loadedDescriptor(String modelId) {
        return new ModelDescriptor(modelId, "llama", 22, 2048, 32000, 32, 4L * 1024 * 1024 * 1024,
                QuantizationType.Q4_K_M, "/models/" + modelId, ModelStatus.LOADED, Instant.now());
    }

    @Test
    @DisplayName("no model loaded at all → error, no fallback possible")
    void noModelLoaded() {
        var res = ModelIdResolver.resolve(registry, "anything");

        assertThat(res.isError()).isTrue();
        assertThat(res.errorMessage()).isEqualTo("No model is currently loaded");
    }

    @Test
    @DisplayName("blank/null requested id falls back to the loaded model, no warning")
    void blankRequestedFallsBackSilently() {
        registry.putLoaded(loadedDescriptor("llava-phi-3-mini-int4.gguf"));

        var byNull = ModelIdResolver.resolve(registry, null);
        var byBlank = ModelIdResolver.resolve(registry, "   ");

        assertThat(byNull.isError()).isFalse();
        assertThat(byNull.modelId()).isEqualTo("llava-phi-3-mini-int4.gguf");
        assertThat(byNull.warning()).isNull();

        assertThat(byBlank.modelId()).isEqualTo("llava-phi-3-mini-int4.gguf");
        assertThat(byBlank.warning()).isNull();
    }

    @Test
    @DisplayName("exact match → resolved silently, no warning")
    void exactMatchNoWarning() {
        registry.putLoaded(loadedDescriptor("llava-phi-3-mini-int4.gguf"));

        var res = ModelIdResolver.resolve(registry, "llava-phi-3-mini-int4.gguf");

        assertThat(res.isError()).isFalse();
        assertThat(res.modelId()).isEqualTo("llava-phi-3-mini-int4.gguf");
        assertThat(res.warning()).isNull();
    }

    @Test
    @DisplayName("mismatched name with exactly one model loaded and SINGLE_MODEL_FALLBACK policy → "
            + "falls back WITH a warning (regression: this used to be a hard 503, e.g. curl copied "
            + "from generic docs using \"llava-v1.5-7b\" while the actually loaded model was a different GGUF)")
    void mismatchWithSingleModelFallsBackWithWarning() {
        registry.putLoaded(loadedDescriptor("llava-phi-3-mini-int4.gguf"));

        var res = ModelIdResolver.resolve(registry, "llava-v1.5-7b", ModelIdResolver.FallbackPolicy.SINGLE_MODEL_FALLBACK);

        assertThat(res.isError()).isFalse();
        assertThat(res.modelId()).isEqualTo("llava-phi-3-mini-int4.gguf");
        assertThat(res.warning()).isNotNull().contains("llava-v1.5-7b").contains("llava-phi-3-mini-int4.gguf");
    }

    @Test
    @DisplayName("mismatched name with exactly one model loaded and STRICT policy (the default) → still an error "
            + "(regression guard: InferenceApiServer's native /v1/inference API relies on this exact contract — "
            + "an explicitly-requested nonexistent model id must not be silently substituted, even with one model loaded)")
    void mismatchWithSingleModelAndStrictPolicyIsStillAnError() {
        registry.putLoaded(loadedDescriptor("tinyllama"));

        var viaExplicitStrict = ModelIdResolver.resolve(registry, "does-not-exist", ModelIdResolver.FallbackPolicy.STRICT);
        var viaTwoArgDefault = ModelIdResolver.resolve(registry, "does-not-exist");

        assertThat(viaExplicitStrict.isError()).isTrue();
        assertThat(viaExplicitStrict.errorMessage()).contains("does-not-exist").contains("tinyllama");

        assertThat(viaTwoArgDefault.isError()).isTrue();
        assertThat(viaTwoArgDefault.errorMessage()).contains("does-not-exist").contains("tinyllama");
    }

    @Test
    @DisplayName("mismatched name with multiple models loaded → error listing what's actually loaded, "
            + "regardless of policy (ambiguous — silent fallback is never safe with >1 model loaded)")
    void mismatchWithMultipleModelsIsAnErrorRegardlessOfPolicy() {
        registry.putLoaded(loadedDescriptor("model-a.gguf"));
        registry.putLoaded(loadedDescriptor("model-b.gguf"));

        var strict = ModelIdResolver.resolve(registry, "model-c.gguf", ModelIdResolver.FallbackPolicy.STRICT);
        var fallback = ModelIdResolver.resolve(registry, "model-c.gguf",
                ModelIdResolver.FallbackPolicy.SINGLE_MODEL_FALLBACK);

        assertThat(strict.isError()).isTrue();
        assertThat(strict.errorMessage()).contains("model-c.gguf").contains("model-a.gguf").contains("model-b.gguf");

        assertThat(fallback.isError()).isTrue();
        assertThat(fallback.errorMessage()).contains("model-c.gguf").contains("model-a.gguf").contains("model-b.gguf");
    }

    @Test
    @DisplayName("a model registered but not LOADED (e.g. still LOADING) is not offered as a fallback")
    void nonLoadedModelIsNotAFallbackTarget() {
        NodeDescriptor node = new NodeDescriptor("n1", "192.168.1.1", 9092, 4L * 1024 * 1024 * 1024,
                4L * 1024 * 1024 * 1024, NodeStatus.READY, 0.9, Instant.now(), Instant.now());
        registry.register(
                ModelDescriptor.of("loading-model", "llama", 22, 2048, 32000, 32, QuantizationType.Q4_K_M,
                        "/models/loading-model.gguf"),
                java.util.List.of(node));

        var res = ModelIdResolver.resolve(registry, "anything");

        assertThat(res.isError()).isTrue();
        assertThat(res.errorMessage()).isEqualTo("No model is currently loaded");
    }

    @Test
    @DisplayName("null registry is rejected")
    void nullRegistryRejected() {
        assertThatThrownBy(() -> ModelIdResolver.resolve(null, "x")).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("null policy is rejected")
    void nullPolicyRejected() {
        registry.putLoaded(loadedDescriptor("tinyllama"));
        assertThatThrownBy(() -> ModelIdResolver.resolve(registry, "tinyllama", null))
                .isInstanceOf(IllegalArgumentException.class);
    }
}