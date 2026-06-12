package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Compares quantized row-range matVec on real Phi-3.5 QKV weights against eager dequant.
 */
class Phi3RealQkvMatVecTest {

	private static boolean phiModelPresent() {
		return Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists()
				|| Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf").toFile().exists();
	}

	private static Path phiModelPath() {
		Path p = Path.of("models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
		return p.toFile().exists() ? p : Path.of("../models/Phi-3.5-mini-instruct-Q4_K_M.gguf");
	}

	@Test
	@EnabledIf("phiModelPresent")
	void blk0_attnQkv_quantMatchesDequant() throws Exception {
		try (GgufReader r = GgufReader.open(phiModelPath())) {
			LlamaConfig cfg = LlamaConfig.from(r);
			int H = cfg.hiddenDim();
			int kvDim = cfg.kvDim();
			float[] x = new float[H];
			for (int i = 0; i < H; i++)
				x[i] = (float) Math.sin(i * 0.01);

			GgufReader.QuantizedTensor qt = r.tensorRaw("blk.0.attn_qkv.weight");
			float[] deq = r.tensor("blk.0.attn_qkv.weight");

			float[] qQuant = LlamaTransformerHandler.matVec(qt, x, 0, H, H);
			float[] qDeq = rowRangeMatVec(deq, x, 0, H, H);
			float[] kQuant = LlamaTransformerHandler.matVec(qt, x, H, H + kvDim, H);
			float[] kDeq = rowRangeMatVec(deq, x, H, kvDim, H);

			float maxQ = 0, maxK = 0;
			for (int i = 0; i < H; i++)
				maxQ = Math.max(maxQ, Math.abs(qQuant[i] - qDeq[i]));
			for (int i = 0; i < kvDim; i++)
				maxK = Math.max(maxK, Math.abs(kQuant[i] - kDeq[i]));
			System.out.printf("max|Q_quant-Q_deq|=%.4f max|K_quant-K_deq|=%.4f%n", maxQ, maxK);
			assertThat(maxQ).isLessThan(0.15f);
			assertThat(maxK).isLessThan(0.15f);
		}
	}

	private static float[] rowRangeMatVec(float[] matrix, float[] x, int rowStart, int nRows, int cols) {
		float[] slice = new float[nRows * cols];
		System.arraycopy(matrix, rowStart * cols, slice, 0, slice.length);
		return LlamaTransformerHandler.matVec(slice, x, nRows, cols);
	}
}
