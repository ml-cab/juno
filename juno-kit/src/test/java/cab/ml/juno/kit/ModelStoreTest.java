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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.sun.net.httpserver.HttpServer;

import cab.ml.juno.player.HfGgufFetcher;

/** Lookup, SHA-256 verification and pinned download, against a local server: no network. */
class ModelStoreTest {

	private static final byte[] GOOD = "the real model bytes".getBytes(StandardCharsets.UTF_8);
	private HttpServer server;

	@AfterEach
	void stop() {
		if (server != null)
			server.stop(0);
	}

	private HfGgufFetcher serving(byte[] body) throws IOException {
		server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		server.createContext("/acme/m/resolve/abc123/m.gguf", ex -> {
			ex.sendResponseHeaders(200, body.length);
			try (OutputStream os = ex.getResponseBody()) {
				os.write(body);
			}
		});
		server.start();
		return new HfGgufFetcher(HttpClient.newHttpClient(), "http://127.0.0.1:" + server.getAddress().getPort(),
				Path.of("unused"));
	}

	private static CatalogModel model() throws Exception {
		String sha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(GOOD));
		return new CatalogModel("m", "test model", "acme/m", "abc123", "m.gguf", GOOD.length, sha, "1g");
	}

	@Test
	void aLocalCopyWithTheWrongHashIsReplacedByAVerifiedDownload(@TempDir Path tmp) throws Exception {
		Path models = Files.createDirectories(tmp.resolve("models"));
		byte[] wrong = new byte[GOOD.length];
		Files.write(models.resolve("m.gguf"), wrong);
		Path cache = tmp.resolve("cache");
		ModelStore store = new ModelStore(models, cache, serving(GOOD));

		assertThat(store.present(model())).isEqualTo(models.resolve("m.gguf"));
		Path got = store.ensure(model(), new PrintStream(new ByteArrayOutputStream()));

		assertThat(got).isEqualTo(cache.resolve("m.gguf"));
		assertThat(Files.readAllBytes(got)).isEqualTo(GOOD);
		assertThat(Files.readAllBytes(models.resolve("m.gguf"))).isEqualTo(wrong);
		assertThat(Files.readString(cache.resolve(ModelStore.VERIFIED_FILE))).contains(model().sha256());
	}

	@Test
	void aDownloadWhoseHashIsWrongIsRefusedAndDeleted(@TempDir Path tmp) throws Exception {
		Path cache = tmp.resolve("cache");
		ModelStore store = new ModelStore(null, cache, serving("tampered model bytes".getBytes(StandardCharsets.UTF_8)));

		assertThatThrownBy(() -> store.ensure(model(), new PrintStream(new ByteArrayOutputStream())))
				.isInstanceOf(IOException.class).hasMessageContaining("SHA-256");
		assertThat(cache.resolve("m.gguf")).doesNotExist();
	}

	@Test
	void aVerifiedPresentFileIsUsedWithoutDownloading(@TempDir Path tmp) throws Exception {
		Path models = Files.createDirectories(tmp.resolve("models"));
		Files.write(models.resolve("m.gguf"), GOOD);
		ModelStore store = new ModelStore(models, tmp.resolve("cache"), serving(new byte[0]));

		assertThat(store.ensure(model(), new PrintStream(new ByteArrayOutputStream())))
				.isEqualTo(models.resolve("m.gguf"));
		assertThat(tmp.resolve("cache").resolve("m.gguf")).doesNotExist();
	}
}
