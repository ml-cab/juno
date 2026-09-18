/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.coordinator;

import java.util.Arrays;

import cab.ml.juno.node.InferencePipeline;

/**
 * Draft-simple speculative-decoding draft source ({@code --spec-type
 * draft-simple}): a smaller, independently-loaded GGUF model proposes tokens
 * by greedily decoding itself, continuing from the target model's own
 * context; {@link GenerationLoop} verifies the proposal against the target
 * exactly as it does for {@link NgramDraftCache}.
 *
 * <h3>KV lockstep</h3>
 * This session owns a persistent KV cache slot in {@code draftPipeline},
 * keyed by {@code draftKvKey} (distinct from the target's own kvKey — no
 * collision risk). {@link #propose} always advances that KV by exactly
 * {@code maxDraft} positions with the draft model's own greedy guesses,
 * speculatively. Those guesses may later turn out wrong (the target
 * diverges from them), so {@link #observe} reconciles the draft's KV against
 * the now-known ground truth after every round: it walks forward from the
 * last position both sides are known to agree on, and on the first
 * disagreement (or once it reaches the frontier {@link #propose} advanced
 * to, whichever comes first) issues exactly one corrective forward call
 * feeding the real, ground-truth token at that position — overwriting
 * whatever the draft's own KV held there, byte for byte the same
 * overwrite-in-place semantics {@link InferencePipeline#verifyDraft}
 * documents for the target's own verify window. Any further positions the
 * draft speculatively advanced past that point are simply abandoned: their
 * KV entries are stale but harmless, since the next {@link #propose} call
 * starts from the freshly-corrected frontier and will overwrite them again
 * once it reaches them.
 *
 * <p>Because {@link GenerationLoop#generate} compares the draft's proposal
 * against the target's own (independently computed) prediction and always
 * emits the target's prediction, a bug in this reconciliation can only ever
 * hurt the acceptance rate (worse speedup) — it cannot corrupt the emitted
 * token sequence, which depends solely on the target model's own KV/sampler.
 *
 * <p>Rebuilt fresh for every {@link GenerationLoop#generate} call — including
 * repeated turns of the same chat session — the same way {@link
 * NgramDraftCache} is; the draft model does not yet share the target's
 * cross-turn prefix-cache reuse. This is a known, named perf cost for
 * multi-turn sessions, not a correctness gap: see {@code
 * PLAN-Infra-Tier12.md}.
 */
final class DraftModelSession implements DraftProposer {

	private final InferencePipeline draftPipeline;
	private final String draftKvKey;

	/** {@code -1} until the first {@link #propose} call primes the session. */
	private int draftPos = -1;
	/** Position up to which {@link #draftTokens} is known to match ground truth. */
	private int confirmedPos = -1;
	/** The draft model's own token view; {@code length == draftPos + 1} once primed. */
	private int[] draftTokens = EMPTY;

	DraftModelSession(InferencePipeline draftPipeline, String draftKvKey) {
		this.draftPipeline = draftPipeline;
		this.draftKvKey = draftKvKey;
	}

	@Override
	public int[] propose(int[] allTokens, int maxDraft) {
		if (maxDraft <= 0)
			return EMPTY;
		if (draftPos < 0) {
			// First call: prime the draft model's KV over everything already
			// committed, leaving the most recent token unconsumed — exactly the
			// same "step 0 covers the last prompt token" convention GenerationLoop
			// itself uses for the target's startPos.
			if (allTokens.length > 1) {
				draftPipeline.prefillBatch(draftKvKey, Arrays.copyOfRange(allTokens, 0, allTokens.length - 1), 0);
			}
			draftTokens = allTokens.clone();
			draftPos = allTokens.length - 1;
			confirmedPos = draftPos;
		}

		int[] proposal = new int[maxDraft];
		int[] local = draftTokens;
		int pos = draftPos;
		for (int i = 0; i < maxDraft; i++) {
			float[] logits = draftPipeline.forward(draftKvKey, local, pos);
			int next = argmax(logits);
			proposal[i] = next;
			local = GenerationLoop.appendToken(local, next);
			pos++;
		}
		draftTokens = local;
		draftPos = pos;
		return proposal;
	}

	@Override
	public void observe(int[] allTokens) {
		if (draftPos < 0)
			return; // never primed — propose() was never called with a non-empty budget
		int target = allTokens.length - 1;
		if (target <= confirmedPos)
			return; // nothing new confirmed since the last observe()

		int limit = Math.min(draftPos, target);
		int agree = confirmedPos;
		while (agree < limit && draftTokens[agree + 1] == allTokens[agree + 1])
			agree++;

		if (agree == target) {
			// Everything ground truth knows about is already correctly reflected
			// in the draft's own tentative KV — nothing to resync.
			confirmedPos = agree;
			return;
		}

		// Diverged at agree+1 (or the draft's tentative frontier never reached that
		// far) — overwrite that position's KV with the real token. Everything the
		// draft speculatively wrote beyond agree+1 is now stale and abandoned; the
		// next propose() call starts from here and will overwrite it again once
		// (if ever) it reaches those positions.
		int realToken = allTokens[agree + 1];
		int[] resyncWindow = GenerationLoop.appendToken(Arrays.copyOf(draftTokens, agree + 1), realToken);
		draftPipeline.forward(draftKvKey, resyncWindow, agree + 1);
		draftTokens = resyncWindow;
		draftPos = agree + 1;
		confirmedPos = agree + 1;
	}

	@Override
	public void close() {
		if (draftPos >= 0)
			draftPipeline.evict(draftKvKey);
	}

	private static int argmax(float[] logits) {
		int best = 0;
		for (int i = 1; i < logits.length; i++)
			if (logits[i] > logits[best])
				best = i;
		return best;
	}

	private static final int[] EMPTY = new int[0];
}
