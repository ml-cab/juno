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
package cab.ml.juno.lora;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Parses the {@code --lora-play} CLI/env syntax:
 * {@code path[:scale][,path[:scale]]*}. A bare path (no colon) defaults to
 * scale {@code 1.0}, so the original single-file usage is unchanged.
 *
 * <p>
 * The scale suffix is only recognized as the text after the <em>last</em>
 * colon in an entry, and only when that text parses as a finite float — so a
 * Windows drive-letter path such as {@code C:\models\a.lora} is not misread
 * as path {@code C} with scale {@code \models\a.lora}; it falls back to
 * scale {@code 1.0} for the whole string, and
 * {@code C:\models\a.lora:0.5} still finds the trailing {@code 0.5}.
 *
 * <p>
 * Multiple entries are comma-separated: {@code a.lora:0.5,b.lora:1.0}.
 */
public final class LoraPlaySpec {

	/** One parsed {@code --lora-play} entry. */
	public record Entry(Path path, float scale) {
		public Entry {
			Objects.requireNonNull(path, "path");
			if (!Float.isFinite(scale))
				throw new IllegalArgumentException("scale must be finite: " + scale);
		}
	}

	private LoraPlaySpec() {
	}

	public static List<Entry> parse(String raw) {
		if (raw == null || raw.isBlank())
			throw new IllegalArgumentException("--lora-play requires at least one adapter path");
		List<Entry> entries = new ArrayList<>();
		for (String part : raw.split(",")) {
			String s = part.strip();
			if (s.isEmpty())
				throw new IllegalArgumentException("--lora-play has an empty entry in: " + raw);
			entries.add(parseEntry(s));
		}
		return entries;
	}

	private static Entry parseEntry(String s) {
		int lastColon = s.lastIndexOf(':');
		if (lastColon > 0 && lastColon < s.length() - 1) {
			Float scale = tryParseFloat(s.substring(lastColon + 1));
			if (scale != null)
				return new Entry(Path.of(s.substring(0, lastColon)), scale);
		}
		return new Entry(Path.of(s), 1.0f);
	}

	private static Float tryParseFloat(String s) {
		try {
			return Float.parseFloat(s);
		} catch (NumberFormatException e) {
			return null;
		}
	}

	/** Loads every entry and folds them into one playback-ready {@link LoraAdapterSet}. */
	public static LoraAdapterSet loadAndMerge(String raw) throws IOException {
		List<Entry> entries = parse(raw);
		List<LoraPlaybackMerge.ScaledAdapterSet> sets = new ArrayList<>(entries.size());
		for (Entry e : entries)
			sets.add(new LoraPlaybackMerge.ScaledAdapterSet(LoraAdapterSet.load(e.path()), e.scale(),
					e.path().toString()));
		return LoraPlaybackMerge.merge(sets);
	}
}
