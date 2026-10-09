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

import java.util.List;

import cab.ml.juno.node.InferencePipeline;
import cab.ml.juno.tokenizer.ChatMessage;
import cab.ml.juno.tokenizer.ChatTemplateFormatter;
import cab.ml.juno.tokenizer.Tokenizer;

/**
 * One request's view of the context limit. Without context shifting it changes
 * nothing: positions pass through and the pipeline fails at the limit as it
 * always has. With context shifting, before any forward pass that would write
 * past {@link InferencePipeline#contextLimit()} it drops the oldest positions
 * after the kept prefix (the system prompt) from the request's KV, half of what
 * lies after the prefix, and later positions move down by that many. Callers
 * keep counting positions as if nothing had been dropped; {@link #place} turns
 * that count into the position to write.
 *
 * <p>Requests are refused up front where a shift could not work: a pipeline that
 * cannot shift, draft-model speculation (its own KV would no longer line up with
 * the target's), and a system prompt that fills half the context or more.
 */
final class ContextWindow {

	private static final ContextWindow OFF = new ContextWindow(false, Integer.MAX_VALUE, 0);

	private final boolean enabled;
	private final int limit;
	private final int keep;
	private int shifted;
	private boolean truncated;

	private ContextWindow(boolean enabled, int limit, int keep) {
		this.enabled = enabled;
		this.limit = limit;
		this.keep = keep;
	}

	/**
	 * @param promptIds the request's encoded prompt, as the loop formatted it
	 * @throws IllegalArgumentException when the request asks for context shifting
	 *                                  and it cannot work here
	 */
	static ContextWindow open(InferenceRequest request, int[] promptIds, Tokenizer tokenizer,
			InferencePipeline pipeline, SpeculativeDecodeOptions spec) {
		if (!ContextShiftOptions.enabledFor(request))
			return OFF;
		if (!pipeline.supportsContextShift())
			throw new IllegalArgumentException("context shift was requested, but this deployment cannot shift a"
					+ " request's KV cache (" + pipeline.getClass().getSimpleName() + "); it is available in local mode");
		if (spec != null && spec.specType() == SpeculativeDecodeOptions.SpecType.DRAFT_SIMPLE)
			throw new IllegalArgumentException("context shift cannot be combined with --spec-type draft-simple: the"
					+ " draft model's KV cache would no longer line up with the target model's after a shift");
		int limit = pipeline.contextLimit();
		int keep = keepTokens(tokenizer, request, promptIds);
		if (keep > limit / 2)
			throw new IllegalArgumentException("context shift keeps the system prompt in place, and at " + keep
					+ " tokens it fills more than half of the " + limit + "-token context; shorten the system prompt"
					+ " or turn context shift off");
		return new ContextWindow(true, limit, keep);
	}

	/**
	 * Tokens kept in place by every shift: those the formatted conversation shares
	 * with its leading system messages followed by an empty user turn, or just the
	 * BOS token when there is no system message. The empty user turn is what makes
	 * this hold on templates that fold the system text into the first user turn
	 * (Mistral, Phi-3), where the system messages formatted alone render none of it.
	 */
	static int keepTokens(Tokenizer tokenizer, InferenceRequest request, int[] promptIds) {
		List<ChatMessage> messages = request.messages();
		int n = 0;
		while (n < messages.size() && "system".equals(messages.get(n).role()))
			n++;
		if (n == 0)
			return promptIds.length > 0 && promptIds[0] == tokenizer.bosTokenId() ? 1 : 0;
		List<ChatMessage> head = new java.util.ArrayList<>(messages.subList(0, n));
		head.add(ChatMessage.user(""));
		String system = ChatTemplateFormatter.forModelType(request.modelId()).format(head);
		int[] sys = tokenizer.encode(system);
		int common = 0;
		while (common < sys.length && common < promptIds.length && sys[common] == promptIds[common])
			common++;
		return common;
	}

	boolean enabled() {
		return enabled;
	}

	/**
	 * The prompt to prefill. Unchanged unless shifting is on and the prompt does
	 * not fit the context; then the kept prefix plus the latest tokens, filling
	 * half of what the context holds after the prefix.
	 */
	int[] fitPrompt(int[] promptIds) {
		if (!enabled || promptIds.length <= limit)
			return promptIds;
		int tail = (limit - keep) / 2;
		int[] out = new int[keep + tail];
		System.arraycopy(promptIds, 0, out, 0, keep);
		System.arraycopy(promptIds, promptIds.length - tail, out, keep, tail);
		truncated = true;
		return out;
	}

	/**
	 * Position to write a forward pass of {@code width} rows at, for a request that
	 * has counted {@code logicalPos} positions so far. Shifts the request's KV first
	 * when the rows would cross the limit.
	 */
	int place(InferencePipeline pipeline, String kvKey, int logicalPos, int width) {
		int pos = logicalPos - shifted;
		if (!enabled || pos + width <= limit)
			return pos;
		int discard = Math.max(pos + width - limit, (pos - keep) / 2);
		if (discard < 1 || keep + discard > pos)
			throw new IllegalStateException("context shift cannot make room for " + width + " positions at " + pos
					+ " with " + keep + " kept in a " + limit + "-token context");
		ContextShiftEvent ev = new ContextShiftEvent();
		ev.begin();
		pipeline.shiftKv(kvKey, pos, keep, discard);
		if (ev.shouldCommit()) {
			ev.requestId = kvKey;
			ev.seqLen = pos;
			ev.keep = keep;
			ev.discard = discard;
			ev.commit();
		}
		shifted += discard;
		return pos - discard;
	}

	/**
	 * Whether the request's KV no longer lines up with its prompt tokens (it was
	 * shifted, or the prompt was cut), so a session must not offer it as a prefix.
	 */
	boolean movedKv() {
		return shifted > 0 || truncated;
	}
}
