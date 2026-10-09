package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The CPU attention's window: a row at {@code seqLen} attends over keys
 * {@code [max(0, seqLen - window), seqLen)}, {@code window} 0 meaning none, the
 * same contract as the GPU kernel. Held bit-exact, since the windowed loop must
 * run the same arithmetic over fewer keys.
 */
@DisplayName("GqaMath - attention window")
class GqaMathWindowTest {

	private static final int HEADS = 8;
	private static final int KV_HEADS = 2;
	private static final int HEAD_DIM = 16;
	private static final int KV_DIM = KV_HEADS * HEAD_DIM;
	private static final int GQA = HEADS / KV_HEADS;

	private static float[] random(int n, Random rng) {
		float[] a = new float[n];
		for (int i = 0; i < n; i++)
			a[i] = (float) rng.nextGaussian();
		return a;
	}

	private static float[] attend(float[] q, float[] k, float[] v, int seqLen, int window) {
		float[] out = new float[HEADS * HEAD_DIM];
		GqaMath.attend(q, k, v, seqLen, out, new float[seqLen], HEADS, HEAD_DIM, GQA, KV_DIM, window);
		return out;
	}

	/** The loop as it stood before the window: every key in {@code [0, seqLen)}. */
	private static float[] unwindowedReference(float[] q, float[] k, float[] v, int seqLen) {
		float[] out = new float[HEADS * HEAD_DIM];
		float[] scores = new float[seqLen];
		float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
		for (int h = 0; h < HEADS; h++) {
			int kBase = (h / GQA) * HEAD_DIM;
			int qBase = h * HEAD_DIM;
			for (int t = 0; t < seqLen; t++) {
				float dot = 0f;
				for (int d = 0; d < HEAD_DIM; d++)
					dot += q[qBase + d] * k[t * KV_DIM + kBase + d];
				scores[t] = dot * scale;
			}
			LlamaTransformerHandler.softmax(scores, seqLen);
			for (int t = 0; t < seqLen; t++)
				for (int d = 0; d < HEAD_DIM; d++)
					out[qBase + d] += scores[t] * v[t * KV_DIM + kBase + d];
		}
		return out;
	}

	@Test
	@DisplayName("window 0, and any window no shorter than the context, are bit-identical to no window")
	void noWindowIsUnchanged() {
		Random rng = new Random(1);
		for (int seqLen = 1; seqLen <= 40; seqLen++) {
			float[] q = random(HEADS * HEAD_DIM, rng);
			float[] k = random(seqLen * KV_DIM, rng);
			float[] v = random(seqLen * KV_DIM, rng);
			float[] ref = unwindowedReference(q, k, v, seqLen);
			assertThat(attend(q, k, v, seqLen, 0)).as("window 0, seqLen " + seqLen).containsExactly(ref);
			assertThat(attend(q, k, v, seqLen, seqLen)).as("window = seqLen " + seqLen).containsExactly(ref);
			assertThat(attend(q, k, v, seqLen, 262144)).as("window 262144, seqLen " + seqLen).containsExactly(ref);
		}
	}

	@Test
	@DisplayName("a window attends over exactly the last rows: equal to attending over those rows alone")
	void windowIsTheLastRows() {
		Random rng = new Random(2);
		int seqLen = 37;
		float[] q = random(HEADS * HEAD_DIM, rng);
		float[] k = random(seqLen * KV_DIM, rng);
		float[] v = random(seqLen * KV_DIM, rng);
		for (int window : new int[] { 1, 2, 5, 16, 36 }) {
			int lo = seqLen - window;
			float[] kTail = Arrays.copyOfRange(k, lo * KV_DIM, seqLen * KV_DIM);
			float[] vTail = Arrays.copyOfRange(v, lo * KV_DIM, seqLen * KV_DIM);
			assertThat(attend(q, k, v, seqLen, window)).as("window " + window)
					.containsExactly(unwindowedReference(q, kTail, vTail, window));
			assertThat(attend(q, k, v, seqLen, window)).as("window " + window + " differs from none")
					.isNotEqualTo(attend(q, k, v, seqLen, 0));
		}
	}

	@Test
	@DisplayName("rows outside the window never reach the output (NaN there leaves it unchanged)")
	void rowsOutsideTheWindowAreNotRead() {
		Random rng = new Random(3);
		int seqLen = 30;
		int window = 9;
		float[] q = random(HEADS * HEAD_DIM, rng);
		float[] k = random(seqLen * KV_DIM, rng);
		float[] v = random(seqLen * KV_DIM, rng);
		float[] clean = attend(q, k, v, seqLen, window);
		Arrays.fill(k, 0, (seqLen - window) * KV_DIM, Float.NaN);
		Arrays.fill(v, 0, (seqLen - window) * KV_DIM, Float.NaN);
		assertThat(attend(q, k, v, seqLen, window)).containsExactly(clean);
	}
}
