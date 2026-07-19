package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraAdapterSet;

@DisplayName("LoraQvInitializer / LoraInitializer")
class LoraQvInitializerTest {

	private final LlamaConfig cfg = new LlamaConfig(16, 3, 2, 1, 8, 32, 200, 1e-5f, 10000f, "llama");

	@Test
	@DisplayName("qv() creates adapters for every layer on wq and wv")
	void qv_factory_creates_correct_adapters() {
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

	@Test
	@DisplayName("all-linear creates seven projections per layer in stable order")
	void all_linear_count_and_order() {
		LoraAdapterSet set = LoraInitializer.create(cfg, LoraProjection.allLinear(), 2, 2f, new Random(1));
		assertThat(set.size()).isEqualTo(3 * 7);
		var keys = set.asMap().keySet().iterator();
		assertThat(keys.next()).isEqualTo("0:wq");
		assertThat(keys.next()).isEqualTo("0:wk");
		assertThat(keys.next()).isEqualTo("0:wv");
		assertThat(keys.next()).isEqualTo("0:wo");
		assertThat(keys.next()).isEqualTo("0:wgate");
		assertThat(keys.next()).isEqualTo("0:wup");
		assertThat(keys.next()).isEqualTo("0:wdown");
		assertThat(keys.next()).isEqualTo("1:wq");
	}

	@Test
	@DisplayName("GQA K/V dimensions use kvDim")
	void gqa_kv_dimensions() {
		LoraAdapterSet set = LoraInitializer.create(cfg, LoraProjection.parseTargets("wk,wv"), 2, 2f,
				new Random(1));
		assertThat(set.get(0, "wk").outDim).isEqualTo(cfg.kvDim());
		assertThat(set.get(0, "wv").outDim).isEqualTo(cfg.kvDim());
	}

	@Test
	@DisplayName("validate rejects dimension mismatch")
	void validate_rejects_mismatch() {
		LoraAdapterSet set = LoraQvInitializer.qv(cfg, 4, 4f, new Random(5));
		LlamaConfig other = new LlamaConfig(32, 3, 4, 2, 8, 64, 200, 1e-5f, 10000f, "llama");
		assertThatThrownBy(() -> LoraInitializer.validate(set, other)).isInstanceOf(IllegalArgumentException.class);
	}
}
