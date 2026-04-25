package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraAdapterSet;

@DisplayName("LoraQvInitializer")
class LoraQvInitializerTest {

	@Test
	@DisplayName("qv() creates adapters for every layer on wq and wv")
	void qv_factory_creates_correct_adapters() {
		LlamaConfig cfg = new LlamaConfig(16, 3, 2, 1, 8, 32, 200, 1e-5f, 10000f, "llama");
		LoraAdapterSet set = LoraQvInitializer.qv(cfg, 4, 4f, new Random(5));

		assertThat(set.size()).isEqualTo(6);
		for (int li = 0; li < 3; li++) {
			assertThat(set.get(li, "wq")).isNotNull();
			assertThat(set.get(li, "wv")).isNotNull();
			assertThat(set.get(li, "wk")).isNull();
		}
		assertThat(set.get(0, "wq").outDim).isEqualTo(16);
		assertThat(set.get(0, "wq").inDim).isEqualTo(16);
		assertThat(set.get(0, "wv").outDim).isEqualTo(8);
		assertThat(set.get(0, "wv").inDim).isEqualTo(16);
	}
}
