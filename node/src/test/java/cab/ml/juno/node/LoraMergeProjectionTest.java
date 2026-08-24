package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraProjection merge mapping")
class LoraMergeProjectionTest {

	@Test
	@DisplayName("all seven projections map to GGUF suffixes")
	void all_projections_have_suffixes() {
		assertThat(LoraProjection.WQ.ggufSuffix()).isEqualTo("attn_q.weight");
		assertThat(LoraProjection.WK.ggufSuffix()).isEqualTo("attn_k.weight");
		assertThat(LoraProjection.WV.ggufSuffix()).isEqualTo("attn_v.weight");
		assertThat(LoraProjection.WO.ggufSuffix()).isEqualTo("attn_output.weight");
		assertThat(LoraProjection.WGATE.ggufSuffix()).isEqualTo("ffn_gate.weight");
		assertThat(LoraProjection.WUP.ggufSuffix()).isEqualTo("ffn_up.weight");
		assertThat(LoraProjection.WDOWN.ggufSuffix()).isEqualTo("ffn_down.weight");
	}

	@Test
	@DisplayName("unknown projection key is rejected")
	void unknown_key_rejected() {
		assertThatThrownBy(() -> LoraProjection.fromKey("wxyz")).isInstanceOf(IllegalArgumentException.class);
	}
}
