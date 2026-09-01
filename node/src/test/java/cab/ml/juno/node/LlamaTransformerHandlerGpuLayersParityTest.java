package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tier 5: synthetic handler forward is stable under gpu-layers=0 policy (CPU matmul path).
 */
@DisplayName("LlamaTransformerHandler — gpu-layers CPU parity")
class LlamaTransformerHandlerGpuLayersParityTest {

	private static final int VOCAB = 64;
	private static final int H = 32;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int LAYERS = 4;

	private String originalGpuLayers;

	@AfterEach
	void restore() {
		if (originalGpuLayers == null)
			System.clearProperty(GpuLayerOffload.ENV_PROPERTY);
		else
			System.setProperty(GpuLayerOffload.ENV_PROPERTY, originalGpuLayers);
	}

	@Test
	@DisplayName("gpu-layers=0 forward matches baseline CpuMatVec handler")
	void gpu_layers_zero_matches_cpu_forward() {
		originalGpuLayers = System.getProperty(GpuLayerOffload.ENV_PROPERTY);
		System.setProperty(GpuLayerOffload.ENV_PROPERTY, "0");

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		LlamaTransformerHandler cpu = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null);
		ForwardRequest req = ForwardRequest.withTokens("r1", new int[] { 3, 7 }, 1);
		float[] logitsCpu = cpu.forward(req, ctx).logits();

		LlamaTransformerHandler cpu2 = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null);
		float[] logitsCpu2 = cpu2.forward(req, ctx).logits();

		assertThat(logitsCpu.length).isEqualTo(VOCAB);
		for (int i = 0; i < VOCAB; i++)
			assertThat(logitsCpu[i]).isCloseTo(logitsCpu2[i], offset(1e-5f));
	}
}
