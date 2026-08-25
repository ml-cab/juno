package cab.ml.juno.vision;

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

import cab.ml.juno.node.GgufReader;

/**
 * Tests {@link VisionConfig#from}'s selection of pixel normalisation
 * constants — the fix for moondream2 (SigLIP) being fed CLIP-normalised
 * pixels. See {@code docs/Vision-I2T.md}, "Known issues / fixes".
 */
@DisplayName("VisionConfig — CLIP vs SigLIP normalisation defaults")
class VisionConfigNormalizationTest {

	private static final int MAGIC = 0x46554747;
	private static final int TYPE_ARRAY = 9;
	private static final int TYPE_FLOAT32 = 6;

	@Test
	@DisplayName("no v.class_embd tensor (SigLIP-style, e.g. moondream2) -> SigLIP 0.5/0.5/0.5 defaults")
	void no_cls_token_defaults_to_siglip(@TempDir Path tempDir) throws IOException {
		Path gguf = buildGguf(tempDir, false, null, null);

		try (GgufReader r = GgufReader.open(gguf)) {
			VisionConfig cfg = VisionConfig.from(r);
			assertThat(cfg.imageMean()).containsExactly(0.5f, 0.5f, 0.5f);
			assertThat(cfg.imageStd()).containsExactly(0.5f, 0.5f, 0.5f);
		}
	}

	@Test
	@DisplayName("v.class_embd tensor present (CLIP-style, e.g. LLaVA) -> OpenAI CLIP defaults")
	void cls_token_present_defaults_to_clip(@TempDir Path tempDir) throws IOException {
		Path gguf = buildGguf(tempDir, true, null, null);

		try (GgufReader r = GgufReader.open(gguf)) {
			VisionConfig cfg = VisionConfig.from(r);
			assertThat(cfg.imageMean()).containsExactly(0.48145466f, 0.4578275f, 0.40821073f);
			assertThat(cfg.imageStd()).containsExactly(0.26862954f, 0.26130258f, 0.27577711f);
		}
	}

	@Test
	@DisplayName("clip.vision.image_mean/image_std in GGUF metadata override the architecture default")
	void explicit_metadata_overrides_default(@TempDir Path tempDir) throws IOException {
		// No CLS token (would default to SigLIP 0.5/0.5/0.5), but the GGUF
		// explicitly declares different values — those must win.
		float[] explicitMean = { 0.1f, 0.2f, 0.3f };
		float[] explicitStd  = { 0.4f, 0.5f, 0.6f };
		Path gguf = buildGguf(tempDir, false, explicitMean, explicitStd);

		try (GgufReader r = GgufReader.open(gguf)) {
			VisionConfig cfg = VisionConfig.from(r);
			assertThat(cfg.imageMean()).containsExactly(0.1f, 0.2f, 0.3f);
			assertThat(cfg.imageStd()).containsExactly(0.4f, 0.5f, 0.6f);
		}
	}

	// ── Minimal GGUF builder: optional v.class_embd tensor + optional
	// clip.vision.image_mean/image_std metadata arrays ──────────────────────

	private static Path buildGguf(Path dir, boolean withClsToken, float[] mean, float[] std) throws IOException {
		final int ALIGNMENT = 32;

		java.util.List<byte[]> kvChunks = new java.util.ArrayList<>();
		if (mean != null) kvChunks.add(floatArrayKv("clip.vision.image_mean", mean));
		if (std  != null) kvChunks.add(floatArrayKv("clip.vision.image_std", std));
		int kvCount = kvChunks.size();
		int kvBytes = kvChunks.stream().mapToInt(b -> b.length).sum();

		byte[] tensorName = "v.class_embd".getBytes(StandardCharsets.UTF_8);
		int tensorCount = withClsToken ? 1 : 0;
		byte[] tensorData = new byte[4]; // one F32 element
		ByteBuffer.wrap(tensorData).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.0f);

		int infoBytes = withClsToken ? (8 + tensorName.length + 4 + 8 + 4 + 8) : 0;
		int prePad = 24 + kvBytes + infoBytes;
		int aligned = ((prePad + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
		int dataLen = withClsToken ? tensorData.length : 0;

		ByteBuffer buf = ByteBuffer.allocate(aligned + dataLen).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(MAGIC);
		buf.putInt(3);
		buf.putLong(tensorCount);
		buf.putLong(kvCount);
		for (byte[] chunk : kvChunks)
			buf.put(chunk);
		if (withClsToken) {
			buf.putLong(tensorName.length);
			buf.put(tensorName);
			buf.putInt(1);       // ndims
			buf.putLong(1);      // dim[0]
			buf.putInt(0);       // F32
			buf.putLong(0);      // offset
		}
		buf.put(new byte[aligned - prePad]);
		if (withClsToken)
			buf.put(tensorData);

		Path gguf = dir.resolve("vision_config_norm_test_"
				+ withClsToken + "_" + (mean != null) + ".gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}

	private static byte[] floatArrayKv(String key, float[] values) {
		byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
		ByteBuffer buf = ByteBuffer.allocate(8 + keyBytes.length + 4 + 4 + 8 + values.length * 4)
				.order(ByteOrder.LITTLE_ENDIAN);
		buf.putLong(keyBytes.length);
		buf.put(keyBytes);
		buf.putInt(TYPE_ARRAY);
		buf.putInt(TYPE_FLOAT32);
		buf.putLong(values.length);
		for (float v : values)
			buf.putFloat(v);
		return buf.array();
	}
}