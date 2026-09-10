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
package cab.ml.juno.kvcache;

/**
 * CLI / env policy for {@code --kv-page-size} / {@code JUNO_KV_PAGE_SIZE}.
 *
 * <p>Default is {@value #DEFAULT_PAGE_SIZE} tokens per block. Under
 * {@code --schedule static} the value is unused (dense KV); under continuous
 * schedule it sizes {@link KvBlockPool} pages.
 */
public final class KvPageSizeOptions {

	public static final String ENV = "JUNO_KV_PAGE_SIZE";
	public static final int DEFAULT_PAGE_SIZE = 16;

	private final int pageSize;

	private KvPageSizeOptions(int pageSize) {
		if (pageSize < 1)
			throw new IllegalArgumentException("pageSize must be >= 1");
		this.pageSize = pageSize;
	}

	public static KvPageSizeOptions of(int pageSize) {
		return new KvPageSizeOptions(pageSize);
	}

	public static KvPageSizeOptions defaults() {
		return of(DEFAULT_PAGE_SIZE);
	}

	public static KvPageSizeOptions fromEnv() {
		String raw = firstNonBlank(System.getProperty(ENV), System.getenv(ENV));
		if (raw == null)
			return defaults();
		try {
			return of(Integer.parseInt(raw.strip()));
		} catch (NumberFormatException e) {
			throw new IllegalArgumentException("invalid " + ENV + ": " + raw, e);
		}
	}

	private static String firstNonBlank(String a, String b) {
		if (a != null && !a.isBlank())
			return a;
		if (b != null && !b.isBlank())
			return b;
		return null;
	}

	public int pageSize() {
		return pageSize;
	}

	public String policySummary() {
		return "kv-page-size=" + pageSize;
	}
}
