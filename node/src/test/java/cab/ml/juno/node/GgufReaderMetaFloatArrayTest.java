package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@DisplayName("GgufReader.metaFloatArray — FLOAT32 array metadata (image_mean/image_std style)")
class GgufReaderMetaFloatArrayTest {

	private static final int GGUF_MAGIC = 0x46554747;
	private static final int TYPE_FLOAT32 = 6;
	private static final int TYPE_STRING = 8;
	private static final int TYPE_ARRAY = 9;

	@Test
	@DisplayName("reads a 3-element FLOAT32 array (clip.vision.image_mean shape)")
	void reads_float32_array(@TempDir Path tempDir) throws IOException {
		Path gguf = buildGgufWithOneFloatArray(tempDir, "clip.vision.image_mean",
				new float[] { 0.5f, 0.5f, 0.5f });

		try (GgufReader r = GgufReader.open(gguf)) {
			float[] result = r.metaFloatArray("clip.vision.image_mean", new float[] { -1f, -1f, -1f });
			assertThat(result).containsExactly(0.5f, 0.5f, 0.5f);
		}
	}

	@Test
	@DisplayName("returns the default when the key is absent")
	void missing_key_returns_default(@TempDir Path tempDir) throws IOException {
		Path gguf = buildGgufWithOneFloatArray(tempDir, "clip.vision.image_mean",
				new float[] { 0.5f, 0.5f, 0.5f });
		float[] def = { 0.481f, 0.457f, 0.408f };

		try (GgufReader r = GgufReader.open(gguf)) {
			float[] result = r.metaFloatArray("clip.vision.image_std", def);
			assertThat(result).isSameAs(def);
		}
	}

	@Test
	@DisplayName("returns the default when the key exists but is not a numeric array (e.g. a string)")
	void non_array_value_returns_default(@TempDir Path tempDir) throws IOException {
		Path gguf = buildGgufWithOneString(tempDir, "general.architecture", "phi2");
		float[] def = { 1f, 2f, 3f };

		try (GgufReader r = GgufReader.open(gguf)) {
			float[] result = r.metaFloatArray("general.architecture", def);
			assertThat(result).isSameAs(def);
		}
	}

	// ── Minimal GGUF builders (metadata-only, zero tensors) ─────────────────

	private static Path buildGgufWithOneFloatArray(Path dir, String key, float[] values) throws IOException {
		byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
		int size = 24                       // header: magic+version+tensorCount+kvCount
				+ 8 + keyBytes.length        // key length + key
				+ 4                          // value type (ARRAY)
				+ 4 + 8                      // array elem type + array count
				+ values.length * 4;         // array elements (FLOAT32)

		ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(GGUF_MAGIC);
		buf.putInt(3);      // version
		buf.putLong(0);     // tensor count
		buf.putLong(1);     // kv count

		buf.putLong(keyBytes.length);
		buf.put(keyBytes);
		buf.putInt(TYPE_ARRAY);
		buf.putInt(TYPE_FLOAT32);
		buf.putLong(values.length);
		for (float v : values)
			buf.putFloat(v);

		Path gguf = dir.resolve("meta_float_array_test.gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}

	private static Path buildGgufWithOneString(Path dir, String key, String value) throws IOException {
		byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
		byte[] valBytes = value.getBytes(StandardCharsets.UTF_8);
		int size = 24
				+ 8 + keyBytes.length
				+ 4                          // value type (STRING)
				+ 8 + valBytes.length;       // string length + string bytes

		ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(GGUF_MAGIC);
		buf.putInt(3);
		buf.putLong(0);
		buf.putLong(1);

		buf.putLong(keyBytes.length);
		buf.put(keyBytes);
		buf.putInt(TYPE_STRING);
		buf.putLong(valBytes.length);
		buf.put(valBytes);

		Path gguf = dir.resolve("meta_string_test.gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}
}