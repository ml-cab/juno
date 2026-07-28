package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LoraModelLayout — architecture projection bindings")
class LoraModelLayoutTest {

	@Test
	@DisplayName("LLaMA layout uses separate attn_q/k/v and ffn_gate/up tensors")
	void llama_separateTensors() {
		LlamaConfig cfg = LlamaConfig.synthetic(100, 64, 4, 2, 2);
		LoraModelLayout layout = LoraModelLayout.llama(cfg);

		assertThat(layout.architecture()).isEqualTo("llama");
		assertThat(layout.binding(0, LoraProjection.WQ).physicalName()).isEqualTo("blk.0.attn_q.weight");
		assertThat(layout.binding(0, LoraProjection.WK).physicalName()).isEqualTo("blk.0.attn_k.weight");
		assertThat(layout.binding(0, LoraProjection.WV).physicalName()).isEqualTo("blk.0.attn_v.weight");
		assertThat(layout.binding(0, LoraProjection.WO).physicalName()).isEqualTo("blk.0.attn_output.weight");
		assertThat(layout.binding(0, LoraProjection.WGATE).physicalName()).isEqualTo("blk.0.ffn_gate.weight");
		assertThat(layout.binding(0, LoraProjection.WUP).physicalName()).isEqualTo("blk.0.ffn_up.weight");
		assertThat(layout.binding(0, LoraProjection.WDOWN).physicalName()).isEqualTo("blk.0.ffn_down.weight");

		LoraProjectionBinding wq = layout.binding(1, LoraProjection.WQ);
		assertThat(wq.rowOffset()).isEqualTo(0);
		assertThat(wq.rowCount()).isEqualTo(cfg.hiddenDim());
		assertThat(wq.inDim()).isEqualTo(cfg.hiddenDim());
		assertThat(wq.outDim()).isEqualTo(cfg.hiddenDim());

		LoraProjectionBinding wk = layout.binding(0, LoraProjection.WK);
		assertThat(wk.outDim()).isEqualTo(cfg.kvDim());
		assertThat(wk.inDim()).isEqualTo(cfg.hiddenDim());

		LoraProjectionBinding wdown = layout.binding(0, LoraProjection.WDOWN);
		assertThat(wdown.inDim()).isEqualTo(cfg.intermediateSize());
		assertThat(wdown.outDim()).isEqualTo(cfg.hiddenDim());
	}

	@Test
	@DisplayName("Qwen2 layout matches LLaMA physical names and dims")
	void qwen2_matchesLlamaPhysical() {
		LlamaConfig cfg = new LlamaConfig(64, 2, 4, 2, 16, 128, 100, 1e-5f, 10000f, "qwen2");
		LoraModelLayout layout = LoraModelLayout.qwen2(cfg);
		assertThat(layout.architecture()).isEqualTo("qwen2");
		assertThat(layout.binding(0, LoraProjection.WQ).physicalName()).isEqualTo("blk.0.attn_q.weight");
		assertThat(layout.binding(0, LoraProjection.WGATE).physicalName()).isEqualTo("blk.0.ffn_gate.weight");
	}

	@Test
	@DisplayName("Phi-3 layout uses fused QKV and gate/up with correct row ranges")
	void phi3_fusedSlices() {
		LlamaConfig cfg = new LlamaConfig(64, 2, 4, 2, 16, 128, 100, 1e-5f, 10000f, "phi3");
		LoraModelLayout layout = LoraModelLayout.phi3(cfg);
		int H = cfg.hiddenDim();
		int KV = cfg.kvDim();
		int I = cfg.intermediateSize();

		LoraProjectionBinding wq = layout.binding(0, LoraProjection.WQ);
		assertThat(wq.physicalName()).isEqualTo("blk.0.attn_qkv.weight");
		assertThat(wq.rowOffset()).isEqualTo(0);
		assertThat(wq.rowCount()).isEqualTo(H);
		assertThat(wq.outDim()).isEqualTo(H);
		assertThat(wq.inDim()).isEqualTo(H);

		LoraProjectionBinding wk = layout.binding(0, LoraProjection.WK);
		assertThat(wk.physicalName()).isEqualTo("blk.0.attn_qkv.weight");
		assertThat(wk.rowOffset()).isEqualTo(H);
		assertThat(wk.rowCount()).isEqualTo(KV);
		assertThat(wk.outDim()).isEqualTo(KV);

		LoraProjectionBinding wv = layout.binding(0, LoraProjection.WV);
		assertThat(wv.physicalName()).isEqualTo("blk.0.attn_qkv.weight");
		assertThat(wv.rowOffset()).isEqualTo(H + KV);
		assertThat(wv.rowCount()).isEqualTo(KV);

		LoraProjectionBinding wgate = layout.binding(0, LoraProjection.WGATE);
		assertThat(wgate.physicalName()).isEqualTo("blk.0.ffn_up.weight");
		assertThat(wgate.rowOffset()).isEqualTo(0);
		assertThat(wgate.rowCount()).isEqualTo(I);

		LoraProjectionBinding wup = layout.binding(0, LoraProjection.WUP);
		assertThat(wup.physicalName()).isEqualTo("blk.0.ffn_up.weight");
		assertThat(wup.rowOffset()).isEqualTo(I);
		assertThat(wup.rowCount()).isEqualTo(I);

		assertThat(layout.binding(0, LoraProjection.WO).physicalName()).isEqualTo("blk.0.attn_output.weight");
		assertThat(layout.binding(0, LoraProjection.WDOWN).physicalName()).isEqualTo("blk.0.ffn_down.weight");
	}

	@Test
	@DisplayName("Qwen3 dense layout uses qDim for WQ out and WO in")
	void qwen3_qDimShapes() {
		LlamaConfig base = new LlamaConfig(64, 2, 4, 2, 32, 128, 100, 1e-5f, 10000f, "qwen3");
		// headDim=32, numHeads=4 → qDim=128 ≠ hiddenDim=64
		Qwen3Config cfg = new Qwen3Config(base, 0, 0, 0, 1f, false, Qwen3RopeConfig.standard(base));
		assertThat(cfg.qDim()).isEqualTo(128);
		assertThat(cfg.qDim()).isNotEqualTo(cfg.hiddenDim());

		LoraModelLayout layout = LoraModelLayout.qwen3(cfg);
		LoraProjectionBinding wq = layout.binding(0, LoraProjection.WQ);
		assertThat(wq.physicalName()).isEqualTo("blk.0.attn_q.weight");
		assertThat(wq.outDim()).isEqualTo(cfg.qDim());
		assertThat(wq.inDim()).isEqualTo(cfg.hiddenDim());

		LoraProjectionBinding wo = layout.binding(0, LoraProjection.WO);
		assertThat(wo.inDim()).isEqualTo(cfg.qDim());
		assertThat(wo.outDim()).isEqualTo(cfg.hiddenDim());

		LoraProjectionBinding wk = layout.binding(0, LoraProjection.WK);
		assertThat(wk.outDim()).isEqualTo(cfg.kvDim());
	}

	@Test
	@DisplayName("bindingsForPhysical groups Phi QKV adapters onto one tensor")
	void phi3_bindingsForPhysical_groupsSlices() {
		LlamaConfig cfg = new LlamaConfig(64, 1, 4, 2, 16, 128, 100, 1e-5f, 10000f, "phi3");
		LoraModelLayout layout = LoraModelLayout.phi3(cfg);
		var group = layout.bindingsForPhysical("blk.0.attn_qkv.weight");
		assertThat(group).hasSize(3);
		assertThat(group.stream().map(LoraProjectionBinding::projection)).containsExactly(
				LoraProjection.WQ, LoraProjection.WK, LoraProjection.WV);
	}
}
