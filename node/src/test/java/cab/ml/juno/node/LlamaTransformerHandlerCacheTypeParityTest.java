package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.CacheTypeOptions;

/**
 * Default {@code f16} path stays bit-compatible; {@code q8_0} stays within logit tolerance.
 */
@DisplayName("LlamaTransformerHandler — cache-type KV parity")
class LlamaTransformerHandlerCacheTypeParityTest {

	private static final int VOCAB = 64;
	private static final int H = 32;
	private static final int HEADS = 4;
	private static final int KV_HEADS = 2;
	private static final int LAYERS = 4;

	private String prevK;
	private String prevV;

	@AfterEach
	void restore() {
		restoreProp(CacheTypeOptions.ENV_K, prevK);
		restoreProp(CacheTypeOptions.ENV_V, prevV);
	}

	@Test
	@DisplayName("default f16 forward is bit-identical across two handlers")
	void f16_bit_identical() {
		prevK = System.getProperty(CacheTypeOptions.ENV_K);
		prevV = System.getProperty(CacheTypeOptions.ENV_V);
		System.clearProperty(CacheTypeOptions.ENV_K);
		System.clearProperty(CacheTypeOptions.ENV_V);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		ForwardRequest req = ForwardRequest.withTokens("r1", new int[] { 3, 7, 11 }, 1);

		float[] a = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();
		float[] b = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();

		assertThat(a.length).isEqualTo(VOCAB);
		for (int i = 0; i < VOCAB; i++)
			assertThat(a[i]).isEqualTo(b[i]);
	}

	@Test
	@DisplayName("q8_0 logits stay close to f16 baseline")
	void q8_near_f16() {
		prevK = System.getProperty(CacheTypeOptions.ENV_K);
		prevV = System.getProperty(CacheTypeOptions.ENV_V);

		ShardContext ctx = new ShardContext("n0", 0, LAYERS, true, true, VOCAB, H, HEADS);
		ForwardRequest req = ForwardRequest.withTokens("r1", new int[] { 3, 7, 11 }, 1);

		System.setProperty(CacheTypeOptions.ENV_K, "f16");
		System.setProperty(CacheTypeOptions.ENV_V, "f16");
		float[] f16 = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();

		System.setProperty(CacheTypeOptions.ENV_K, "q8_0");
		System.setProperty(CacheTypeOptions.ENV_V, "q8_0");
		float[] q8 = LlamaTransformerHandler.newTestInstance(
				VOCAB, H, HEADS, KV_HEADS, LAYERS, 0, LAYERS, true, true, null)
				.forward(req, ctx).logits();

		assertThat(q8.length).isEqualTo(VOCAB);
		for (int i = 0; i < VOCAB; i++)
			assertThat(q8[i]).isCloseTo(f16[i], offset(0.5f));
	}

	private static void restoreProp(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
