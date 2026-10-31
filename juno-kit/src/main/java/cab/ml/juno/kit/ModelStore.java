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
package cab.ml.juno.kit;


import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;

import cab.ml.juno.player.HfGgufFetcher;

/**
 * Finds catalog models on disk, verifies them by SHA-256 and downloads what is missing. Files are looked up
 * in the tester's {@code --models-dir} (read only) and in the cache directory, where downloads go. A
 * verified hash is remembered per path, size and modification time, so a file is hashed once.
 */
final class ModelStore {

	static final String VERIFIED_FILE = "verified.json";

	record Verified(long size, long modifiedMillis, String sha256) {
	}

	private final List<Path> lookupDirs;
	private final Path cacheDir;
	private final HfGgufFetcher fetcher;
	private final Map<String, Verified> verified;

	ModelStore(Path modelsDir, Path cacheDir, HfGgufFetcher fetcher) throws IOException {
		this.lookupDirs = new ArrayList<>();
		if (modelsDir != null)
			lookupDirs.add(modelsDir);
		lookupDirs.add(cacheDir);
		this.cacheDir = cacheDir;
		this.fetcher = fetcher;
		Path v = cacheDir.resolve(VERIFIED_FILE);
		this.verified = Files.isRegularFile(v)
				? new LinkedHashMap<>(Json.MAPPER.readValue(v.toFile(), new TypeReference<Map<String, Verified>>() {
				}))
				: new LinkedHashMap<>();
	}

	/** The first file of the right name and size, or {@code null}; nothing is hashed or written. */
	Path present(CatalogModel m) {
		for (Path dir : lookupDirs) {
			Path p = dir.resolve(m.file());
			try {
				if (Files.isRegularFile(p) && Files.size(p) == m.bytes())
					return p;
			} catch (IOException ignored) {
			}
		}
		return null;
	}

	/** A verified local copy of {@code m}, downloading it when absent or when the local copy's hash is wrong. */
	Path ensure(CatalogModel m, PrintStream out) throws IOException, InterruptedException {
		Path p = present(m);
		if (p != null) {
			if (hashMatches(p, m, out))
				return p;
			out.println("  " + m.file() + " at " + p + " does not match the catalog's SHA-256; downloading a copy.");
		}
		Files.createDirectories(cacheDir);
		out.println("  Downloading " + m.title() + " (" + (m.bytes() >> 20) + " MiB) from huggingface.co/" + m.repo());
		Path part = cacheDir.resolve(m.file() + ".part");
		Thread ticker = Thread.ofPlatform().daemon().start(() -> {
			try {
				while (true) {
					Thread.sleep(5000);
					long have = Files.exists(part) ? Files.size(part) : 0;
					out.printf("    %s: %d of %d MiB%n", m.file(), have >> 20, m.bytes() >> 20);
				}
			} catch (InterruptedException | IOException ignored) {
			}
		});
		Path got;
		try {
			got = fetcher.downloadPinned(m.repo(), m.revision(), m.file(), cacheDir);
		} finally {
			ticker.interrupt();
		}
		if (!hashMatches(got, m, out)) {
			Files.deleteIfExists(got);
			throw new IOException("downloaded " + m.file() + " does not match the catalog's SHA-256");
		}
		return got;
	}

	private boolean hashMatches(Path p, CatalogModel m, PrintStream out) throws IOException {
		String key = p.toAbsolutePath().normalize().toString();
		long size = Files.size(p);
		long mtime = Files.getLastModifiedTime(p).toMillis();
		Verified v = verified.get(key);
		if (v == null || v.size() != size || v.modifiedMillis() != mtime) {
			out.println("  Verifying " + m.file() + " (" + (size >> 20) + " MiB)");
			v = new Verified(size, mtime, sha256(p));
			verified.put(key, v);
			Files.createDirectories(cacheDir);
			Json.write(cacheDir.resolve(VERIFIED_FILE), verified);
		}
		return v.sha256().equals(m.sha256());
	}

	static String sha256(Path p) throws IOException {
		try (InputStream in = Files.newInputStream(p)) {
			MessageDigest md = MessageDigest.getInstance("SHA-256");
			byte[] buf = new byte[1 << 20];
			int n;
			while ((n = in.read(buf)) > 0)
				md.update(buf, 0, n);
			return HexFormat.of().formatHex(md.digest());
		} catch (NoSuchAlgorithmException e) {
			throw new IllegalStateException(e);
		}
	}
}
