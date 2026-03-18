package cab.ml.juno.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class ModelDescriptorTest {

	private ModelDescriptor tinyllama() {
		return ModelDescriptor.of("tinyllama", "llama", 22, // totalLayers
				2048, // hiddenDim
				32000, // vocabSize
				32, // numHeads
				QuantizationType.Q4_K_M, "/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf");
	}

	@Test
	void vram_per_layer_is_positive() {
		assertThat(tinyllama().vramPerLayerBytes()).isGreaterThan(0);
	}

	@Test
	void vram_per_layer_scales_with_hidden_dim() {
		ModelDescriptor small = ModelDescriptor.of("s", "llama", 32, 2048, 32000, 32, QuantizationType.FP16, "/s");
		ModelDescriptor large = ModelDescriptor.of("l", "llama", 32, 8192, 128000, 64, QuantizationType.FP16, "/l");
		assertThat(large.vramPerLayerBytes()).isGreaterThan(small.vramPerLayerBytes());
	}

	@Test
	void vram_per_layer_is_lower_for_quantized_than_fp16() {
		ModelDescriptor fp16 = ModelDescriptor.of("m", "llama", 32, 4096, 32000, 32, QuantizationType.FP16, "/m");
		ModelDescriptor q4 = ModelDescriptor.of("m", "llama", 32, 4096, 32000, 32, QuantizationType.Q4_K_M, "/m");
		assertThat(q4.vramPerLayerBytes()).isLessThan(fp16.vramPerLayerBytes());
	}

	@Test
	void initial_status_is_unloaded() {
		assertThat(tinyllama().status()).isEqualTo(ModelStatus.UNLOADED);
	}

	@Test
	void with_status_returns_new_record() {
		ModelDescriptor original = tinyllama();
		ModelDescriptor loading = original.withStatus(ModelStatus.LOADING);

		assertThat(loading.status()).isEqualTo(ModelStatus.LOADING);
		assertThat(original.status()).isEqualTo(ModelStatus.UNLOADED); // immutable
	}

	@Test
	void rejects_blank_model_id() {
		assertThatThrownBy(() -> ModelDescriptor.of("", "llama", 32, 4096, 32000, 32, QuantizationType.FP16, "/m"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("modelId");
	}

	@Test
	void rejects_zero_layers() {
		assertThatThrownBy(() -> ModelDescriptor.of("m", "llama", 0, 4096, 32000, 32, QuantizationType.FP16, "/m"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("totalLayers");
	}

	@Test
	void rejects_zero_hidden_dim() {
		assertThatThrownBy(() -> ModelDescriptor.of("m", "llama", 32, 0, 32000, 32, QuantizationType.FP16, "/m"))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("hiddenDim");
	}

	@Test
	void total_vram_estimate_is_layers_times_per_layer() {
		ModelDescriptor m = tinyllama();
		assertThat(m.totalVramBytes()).isEqualTo(m.vramPerLayerBytes() * m.totalLayers());
	}

	@Test
	void human_readable_size_is_not_blank() {
		assertThat(tinyllama().humanReadableSize()).isNotBlank();
	}
}
