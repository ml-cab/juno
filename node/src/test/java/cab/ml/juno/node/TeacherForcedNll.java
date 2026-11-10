package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Teacher-forced negative log-likelihood per position, as
 * {@link ModelQualityNllLiveTest} scores it: token {@code t} at position
 * {@code t}, scored on token {@code t + 1}, through
 * {@link ForwardPassHandler#forwardVerify} in windows of {@link #WINDOW}.
 * Kept per position so a caller can compare ranges of the context.
 */
final class TeacherForcedNll {

	static final int WINDOW = 128;

	private TeacherForcedNll() {
	}

	/** {@code nll[t] = -log p(ids[t + 1] | ids[0..t])}; evicts {@code session} when done. */
	static double[] perPosition(ForwardPassHandler handler, ShardContext ctx, int[] ids, String session) {
		int predictions = ids.length - 1;
		double[] nll = new double[predictions];
		try {
			for (int start = 0; start < predictions; start += WINDOW) {
				int w = Math.min(WINDOW, predictions - start);
				int[] window = Arrays.copyOfRange(ids, start, start + w);
				float[] logits = handler.forwardVerify(BatchForwardRequest.withTokens(session, window, start), ctx)
						.logits();
				int vocab = logits.length / w;
				for (int i = 0; i < w; i++) {
					int off = i * vocab;
					double max = Double.NEGATIVE_INFINITY;
					for (int v = 0; v < vocab; v++)
						max = Math.max(max, logits[off + v]);
					double sum = 0;
					for (int v = 0; v < vocab; v++)
						sum += Math.exp(logits[off + v] - max);
					nll[start + i] = -(logits[off + ids[start + i + 1]] - max - Math.log(sum));
				}
			}
		} finally {
			handler.evict(session);
		}
		return nll;
	}

	/** Mean of {@code nll[from, to)}. */
	static double mean(double[] nll, int from, int to) {
		double s = 0;
		for (int i = from; i < to; i++)
			s += nll[i];
		return s / (to - from);
	}

	/** The first {@code count} ids of a checked-in id file beside this class. */
	static int[] readIds(String resource, int count) throws IOException {
		try (InputStream in = TeacherForcedNll.class.getResourceAsStream(resource)) {
			assertThat(in).as(resource).isNotNull();
			String s = new String(in.readAllBytes(), StandardCharsets.US_ASCII).strip();
			int[] ids = Arrays.stream(s.split("\\s+")).mapToInt(Integer::parseInt).toArray();
			assertThat(ids.length).as("ids in %s", resource).isGreaterThanOrEqualTo(count);
			return Arrays.copyOf(ids, count);
		}
	}
}
