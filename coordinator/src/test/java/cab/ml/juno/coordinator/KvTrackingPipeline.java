package cab.ml.juno.coordinator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import cab.ml.juno.node.InferencePipeline;

/**
 * Test double that models the one invariant every real handler relies on: a
 * forward pass at position {@code p} under KV key {@code k} reads positions
 * {@code 0..p-1} that were previously written under the same key {@code k}.
 *
 * <p>
 * A real handler that is asked to continue from a position it never wrote
 * silently attends over zeroed KV and produces plausible-looking garbage. This
 * double makes that failure observable: every such call is recorded in
 * {@link #underruns()}, and the returned logits are a function of the tokens
 * actually present in KV, so output produced from missing KV differs from output
 * produced from a correct prefill.
 *
 * <p>
 * The next token is always in {@code [FIRST_TOKEN, FIRST_TOKEN + SPAN)}, which
 * stays clear of the special ids used by {@code SimpleTokenizer}.
 */
final class KvTrackingPipeline implements InferencePipeline {

	static final int VOCAB_SIZE = 1000;
	static final int FIRST_TOKEN = 10;
	static final int SPAN = 190;

	private record Write(String key, int position) {
	}

	private final Map<String, Map<Integer, Integer>> kv = new ConcurrentHashMap<>();
	private final List<String> underruns = Collections.synchronizedList(new ArrayList<>());
	private final List<Write> writeLog = Collections.synchronizedList(new ArrayList<>());

	@Override
	public float[] forward(String requestId, int[] tokens, int startPos) {
		write(requestId, startPos, tokens[startPos]);
		return logitsFor(requestId, startPos);
	}

	@Override
	public void prefillBatch(String requestId, int[] newTokens, int startPosition) {
		for (int p = 0; p < newTokens.length; p++)
			write(requestId, startPosition + p, newTokens[p]);
	}

	@Override
	public int vocabSize() {
		return VOCAB_SIZE;
	}

	@Override
	public void evict(String requestId) {
		kv.remove(requestId);
	}

	private void write(String key, int position, int token) {
		Map<Integer, Integer> positions = kv.computeIfAbsent(key, k -> new ConcurrentHashMap<>());
		for (int p = 0; p < position; p++) {
			if (!positions.containsKey(p)) {
				underruns.add("key=" + key + " forward at position " + position + " but position " + p
						+ " was never written under this key");
				break;
			}
		}
		positions.put(position, token);
		writeLog.add(new Write(key, position));
	}

	private float[] logitsFor(String key, int position) {
		Map<Integer, Integer> positions = kv.get(key);
		long h = 17;
		for (int p = 0; p <= position; p++) {
			Integer t = positions.get(p);
			h = h * 31 + (t == null ? -1 : t);
		}
		int winner = FIRST_TOKEN + (int) Math.floorMod(h, (long) SPAN);
		float[] logits = new float[VOCAB_SIZE];
		logits[winner] = 100.0f;
		return logits;
	}

	/** Every call that continued from a position it had not written under the same key. */
	List<String> underruns() {
		synchronized (underruns) {
			return List.copyOf(underruns);
		}
	}

	/** Opaque marker for {@link #positionsWrittenSince(int, String)}. */
	int mark() {
		return writeLog.size();
	}

	/** Sorted distinct positions written under {@code key} after {@code mark}. */
	List<Integer> positionsWrittenSince(int mark, String key) {
		TreeMap<Integer, Boolean> seen = new TreeMap<>();
		synchronized (writeLog) {
			for (int i = mark; i < writeLog.size(); i++) {
				Write w = writeLog.get(i);
				if (w.key().equals(key))
					seen.put(w.position(), Boolean.TRUE);
			}
		}
		return List.copyOf(seen.keySet());
	}
}
