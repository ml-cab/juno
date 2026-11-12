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

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;
import java.util.logging.Logger;

import cab.ml.juno.kvcache.PrefixCache;
import cab.ml.juno.kvcache.SharedPrefixOptions;
import cab.ml.juno.node.InferencePipeline;

/**
 * Prompts of finished requests, kept as KV so that a later request whose prompt starts the same way (many clients
 * sending one system prompt, two users opening with the same message) starts from that KV instead of computing it.
 *
 * <p>When a request finishes, its prompt's positions are shared into a key of this pool's own
 * ({@link InferencePipeline#shareKvPrefix}): on the paged KV the same pages, held once more; on the dense KV a copy.
 * Nothing writes under a pool key, so a held prompt never changes, and a new request shares only from those keys,
 * never from a live request or session, whose KV its next step writes. Held prompts are released least recently used
 * first once they exceed the budget in positions. Every operation runs under one lock, so a held prompt is not
 * released while a request is being started from it.
 *
 * <p>A request with an image is neither kept nor started from a kept prompt (the vision handler refuses it), and a
 * pipeline that cannot share (remote stages, the LoRA handlers) has no pool.
 */
public final class SharedPrefixPool {

	private static final Logger log = Logger.getLogger(SharedPrefixPool.class.getName());

	/**
	 * The shortest prefix worth sharing. Shorter matches are mostly a chat template's opening tokens, which every
	 * prompt shares and whose prefill costs little.
	 */
	public static final int MIN_SHARED_TOKENS = 16;

	/** Pool keys; a request id is a UUID and a session id never takes this form in practice. */
	static final String KEY_PREFIX = "juno-shared-prefix#";

	private final InferencePipeline pipeline;
	private final int budgetTokens;
	private final PrefixCache trie = new PrefixCache();
	/** Held prompts, key to positions, least recently used first. */
	private final LinkedHashMap<String, Integer> held = new LinkedHashMap<>(16, 0.75f, true);
	private final Object lock = new Object();
	private long heldTokens;
	private long nextKey;
	private final LongAdder hits = new LongAdder();
	private final LongAdder sharedTokens = new LongAdder();

	private SharedPrefixPool(InferencePipeline pipeline, int budgetTokens) {
		this.pipeline = pipeline;
		this.budgetTokens = budgetTokens;
	}

	/** The pool, null when sharing is off, and the startup line saying which and why. */
	public record Setup(SharedPrefixPool pool, String summary) {
	}

	/**
	 * The pool for {@code pipeline}, or none when sharing is off: a budget of 0, or a pipeline that cannot share
	 * while the budget is the default.
	 *
	 * @param loraPlay whether LoRA playback is on (its KV depends on the adapter, so it is never shared)
	 * @throws IllegalArgumentException when a budget above 0 was given explicitly and the pipeline cannot share
	 */
	public static Setup open(InferencePipeline pipeline, SharedPrefixOptions options, boolean loraPlay) {
		long bytesPerToken = pipeline.hostKvBytesPerToken();
		int budget = options.budget(pipeline.contextLimit(), Runtime.getRuntime().maxMemory(), bytesPerToken);
		if (budget == 0)
			return off("--prefix-cache-tokens 0");
		String refusal = loraPlay ? "LoRA playback is on"
				: pipeline.supportsKvPrefixSharing() ? null
						: "this pipeline's KV cannot be shared (a cluster's remote nodes, the LoRA handlers, a"
								+ " quantized host KV beside a GPU attention KV, or a node KV adapter)";
		if (refusal != null) {
			if (options.explicit())
				throw new IllegalArgumentException("--prefix-cache-tokens " + budget + " was given, but " + refusal
						+ "; pass --prefix-cache-tokens 0 or leave it unset");
			return off(refusal);
		}
		String summary = "shared prompt prefixes: on, up to " + budget + " positions of finished requests' prompts"
				+ " kept" + (bytesPerToken > 0 ? " (" + (budget * bytesPerToken >> 20) + " MiB of KV)" : "")
				+ (options.explicit() ? "" : "; default: a quarter of the heap, at most one context");
		log.info(summary);
		return new Setup(new SharedPrefixPool(pipeline, budget), summary);
	}

	private static Setup off(String why) {
		String summary = "shared prompt prefixes: off (" + why + ")";
		log.info(summary);
		return new Setup(null, summary);
	}

	/** For tests: a pool over {@code pipeline} with this budget, whatever the pipeline reports. */
	static SharedPrefixPool of(InferencePipeline pipeline, int budgetTokens) {
		return new SharedPrefixPool(pipeline, budgetTokens);
	}

	/**
	 * Starts {@code kvKey}'s KV from the longest held prompt that {@code promptIds} begins with, up to its last token
	 * (which the request forwards itself for its first logits). Returns the positions shared, 0 when none: nothing
	 * held matches by at least {@link #MIN_SHARED_TOKENS}, or the pipeline refused, in which case {@code kvKey}
	 * holds nothing new.
	 */
	public int shareInto(String kvKey, int[] promptIds) {
		if (promptIds.length - 1 < MIN_SHARED_TOKENS)
			return 0;
		synchronized (lock) {
			PrefixCache.PrefixMatch m = trie.findLongestCommonPrefix(promptIds);
			int len = Math.min(m.matchedTokens(), promptIds.length - 1);
			if (len < MIN_SHARED_TOKENS || !pipeline.shareKvPrefix(m.cacheKey(), kvKey, len))
				return 0;
			held.get(m.cacheKey()); // most recently used
			hits.increment();
			sharedTokens.add(len);
			return len;
		}
	}

	/**
	 * Keeps the prompt {@code kvKey} has just finished with (its first {@code promptIds.length} positions), unless
	 * a held prompt begins with it already, it is shorter than {@link #MIN_SHARED_TOKENS} or longer than the budget, or
	 * the pipeline refuses. Releases the least recently used held prompts until the budget holds. Call it before
	 * {@code kvKey}'s KV is released, while nothing writes it.
	 */
	public void retain(String kvKey, int[] promptIds) {
		int len = promptIds.length;
		if (len < MIN_SHARED_TOKENS || len > budgetTokens)
			return;
		synchronized (lock) {
			PrefixCache.PrefixMatch m = trie.findLongestCommonPrefix(promptIds);
			if (m.matchedTokens() == len) { // held already, alone or as the opening of a longer prompt
				held.get(m.cacheKey());
				return;
			}
			String key = KEY_PREFIX + nextKey++;
			if (!pipeline.shareKvPrefix(kvKey, key, len))
				return;
			trie.cachePrefix(promptIds, len, key);
			held.put(key, len);
			heldTokens += len;
			Iterator<Map.Entry<String, Integer>> eldest = held.entrySet().iterator();
			while (heldTokens > budgetTokens) {
				Map.Entry<String, Integer> e = eldest.next();
				eldest.remove();
				heldTokens -= e.getValue();
				trie.invalidate(e.getKey());
				pipeline.evict(e.getKey());
			}
		}
	}

	/** Requests started from a held prompt since startup. */
	public long hits() {
		return hits.sum();
	}

	/** Positions those requests did not compute. */
	public long sharedTokens() {
		return sharedTokens.sum();
	}

	/** Positions held now. */
	public long heldTokens() {
		synchronized (lock) {
			return heldTokens;
		}
	}

	/** Prompts held now. */
	public int heldPrompts() {
		synchronized (lock) {
			return held.size();
		}
	}

	public int budgetTokens() {
		return budgetTokens;
	}
}
