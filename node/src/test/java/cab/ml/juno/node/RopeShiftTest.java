package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A context shift turns cached keys back by the shift distance instead of
 * recomputing them. For every rotary variant Juno runs, a key rotated for
 * position {@code p} and then shifted back by {@code d} must equal the same key
 * rotated for {@code p - d} directly; any magnitude scale the forward rotation
 * applies (Phi-3's and YaRN's attention factor) must not be applied twice.
 */
@DisplayName("RopeShift")
class RopeShiftTest {

	private static final int HEADS = 3;
	private static final int HEAD_DIM = 16;
	private static final int POS = 3000;
	private static final int DELTA = 1234;

	@FunctionalInterface
	interface Rotate {
		void apply(float[] x, int pos);
	}

	private static void assertShiftMatchesDirect(Rotate forward, RopeShift shift) {
		Random rnd = new Random(7);
		float[] x = new float[HEADS * HEAD_DIM];
		for (int i = 0; i < x.length; i++)
			x[i] = (float) rnd.nextGaussian();
		float[] shifted = x.clone();
		forward.apply(shifted, POS);
		shift.back(DELTA, HEADS).rotate(shifted);
		float[] direct = x.clone();
		forward.apply(direct, POS - DELTA);
		for (int i = 0; i < x.length; i++)
			assertThat(shifted[i]).as("element %d", i).isCloseTo(direct[i], within(2e-3f));
	}

	@Test
	@DisplayName("standard rotation, adjacent pairs (LLaMA family)")
	void standardAdjacent() {
		assertShiftMatchesDirect(
				(x, p) -> LlamaTransformerHandler.rope(x, p, HEADS, HEAD_DIM, 10000f, RopePairing.ADJACENT),
				RopeShift.standard(HEAD_DIM, 10000f, RopePairing.ADJACENT));
	}

	@Test
	@DisplayName("standard rotation, split-half pairs (Qwen2 / Qwen3 without YaRN)")
	void standardSplitHalf() {
		assertShiftMatchesDirect(
				(x, p) -> LlamaTransformerHandler.rope(x, p, HEADS, HEAD_DIM, 1_000_000f, RopePairing.SPLIT_HALF),
				RopeShift.standard(HEAD_DIM, 1_000_000f, RopePairing.SPLIT_HALF));
	}

	@Test
	@DisplayName("Qwen3 YaRN: ramped frequencies, attention factor not applied twice")
	void qwen3Yarn() {
		Qwen3RopeConfig cfg = new Qwen3RopeConfig(1_000_000f, 0.25f, 1.14f, 32768, 131072, true,
				RopePairing.SPLIT_HALF);
		assertShiftMatchesDirect((x, p) -> Qwen3Rope.apply(x, p, HEADS, HEAD_DIM, cfg),
				Qwen3Rope.shift(HEAD_DIM, cfg));
	}

	@Test
	@DisplayName("Phi-3 LongRoPE short factors with attention factor")
	void phi3() {
		float[] shortF = new float[HEAD_DIM / 2];
		float[] longF = new float[HEAD_DIM / 2];
		for (int i = 0; i < shortF.length; i++) {
			shortF[i] = 1f + 0.03f * i;
			longF[i] = 1f + 0.5f * i;
		}
		Phi3RopeConfig cfg = new Phi3RopeConfig(10000f, 1f, 1.19f, 4096, 131072, shortF, longF);
		assertShiftMatchesDirect((x, p) -> Phi3Rope.ropeExt(x, p, HEADS, HEAD_DIM, cfg),
				Phi3Rope.shift(HEAD_DIM, cfg));
	}

	@Test
	@DisplayName("Phi-2 partial rotation: dims past ropeDim untouched")
	void phi2Partial() {
		int ropeDim = HEAD_DIM / 2;
		assertShiftMatchesDirect((x, p) -> Phi2Rope.ropePartial(x, p, HEADS, HEAD_DIM, ropeDim, 10000f),
				RopeShift.partialSplitHalf(HEAD_DIM, ropeDim, 10000f));
	}
}
