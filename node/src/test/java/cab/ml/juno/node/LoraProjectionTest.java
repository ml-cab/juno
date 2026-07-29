package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraProjection")
class LoraProjectionTest {

	private final LlamaConfig cfg = new LlamaConfig(16, 3, 2, 1, 8, 32, 200, 1e-5f, 10000f, "llama");

	@Test
	@DisplayName("qv preset")
	void qv_preset() {
		assertThat(LoraProjection.parseTargets("qv")).containsExactly(LoraProjection.WQ, LoraProjection.WV);
	}

	@Test
	@DisplayName("all / all-linear presets include seven projections in enum order")
	void all_linear_preset() {
		assertThat(LoraProjection.parseTargets("all")).containsExactly(LoraProjection.values());
		assertThat(LoraProjection.parseTargets("all-linear")).containsExactly(LoraProjection.values());
	}

	@Test
	@DisplayName("CSV parsing is case-insensitive and order-preserving")
	void csv_parsing() {
		List<LoraProjection> t = LoraProjection.parseTargets("wo, WGate ,wdown");
		assertThat(t).containsExactly(LoraProjection.WO, LoraProjection.WGATE, LoraProjection.WDOWN);
	}

	@Test
	@DisplayName("dimensions match LlamaConfig for GQA")
	void dimensions() {
		assertThat(LoraProjection.WQ.inDim(cfg)).isEqualTo(16);
		assertThat(LoraProjection.WQ.outDim(cfg)).isEqualTo(16);
		assertThat(LoraProjection.WK.outDim(cfg)).isEqualTo(8);
		assertThat(LoraProjection.WV.outDim(cfg)).isEqualTo(8);
		assertThat(LoraProjection.WO.inDim(cfg)).isEqualTo(16);
		assertThat(LoraProjection.WGATE.outDim(cfg)).isEqualTo(32);
		assertThat(LoraProjection.WUP.outDim(cfg)).isEqualTo(32);
		assertThat(LoraProjection.WDOWN.inDim(cfg)).isEqualTo(32);
		assertThat(LoraProjection.WDOWN.outDim(cfg)).isEqualTo(16);
	}

	@Test
	@DisplayName("GGUF suffix mapping")
	void gguf_mapping() {
		assertThat(LoraProjection.WQ.ggufTensorName(2)).isEqualTo("blk.2.attn_q.weight");
		assertThat(LoraProjection.WGATE.ggufSuffix()).isEqualTo("ffn_gate.weight");
		assertThat(LoraProjection.WUP.ggufSuffix()).isEqualTo("ffn_up.weight");
		assertThat(LoraProjection.WDOWN.ggufSuffix()).isEqualTo("ffn_down.weight");
	}

	@Test
	@DisplayName("unknown, duplicate, and empty targets are rejected")
	void invalid_targets() {
		assertThatThrownBy(() -> LoraProjection.parseTargets("wqq"))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraProjection.parseTargets("wq,wq"))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraProjection.parseTargets(""))
				.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraProjection.parseTargets("wq,"))
				.isInstanceOf(IllegalArgumentException.class);
	}
}
