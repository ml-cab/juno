package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.lora.LoraAdapter;
import cab.ml.juno.lora.LoraAdapterSet;

/**
 * Tests {@link GgufLoraImporter} against synthetic GGUF-LoRA fixtures built
 * on the fly by this test — see {@link GgufLoraImporter}'s javadoc for why
 * there is no real converter-produced fixture available in this environment.
 */
@DisplayName("GgufLoraImporter")
class GgufLoraImporterTest {

	@Test
	@DisplayName("imports lora_a/lora_b pairs into a playback-equivalent LoraAdapterSet")
	void imports_matching_pairs(@TempDir Path dir) throws IOException {
		int layer = 0, inDim = 4, outDim = 3, rank = 2;
		float[] a = randomVector(rank * inDim, 1);
		float[] b = randomVector(outDim * rank, 2);

		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { inDim, rank }, a),
						tensor("blk.0.attn_q.weight.lora_b", new long[] { rank, outDim }, b)),
				OptionalFloat.empty());

		LoraAdapterSet set = GgufLoraImporter.importFrom(gguf, OptionalDouble.empty());
		assertThat(set.size()).isEqualTo(1);
		LoraAdapter adapter = set.get(layer, "wq");
		assertThat(adapter).isNotNull();
		assertThat(adapter.rank).isEqualTo(rank);
		assertThat(adapter.inDim).isEqualTo(inDim);
		assertThat(adapter.outDim).isEqualTo(outDim);
		// no explicit alpha and no metadata alpha -> alpha == rank -> scale 1.0
		assertThat(adapter.scale).isCloseTo(1.0f, within(1e-6f));
		assertThat(adapter.a()).containsExactly(a, within(1e-6f));
		assertThat(adapter.b()).containsExactly(b, within(1e-6f));
	}

	@Test
	@DisplayName("multiple projections and layers import into distinct keys")
	void imports_multiple_keys(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_q.weight.lora_b", new long[] { 2, 3 }, randomVector(6, 2)),
						tensor("blk.1.ffn_down.weight.lora_a", new long[] { 6, 2 }, randomVector(12, 3)),
						tensor("blk.1.ffn_down.weight.lora_b", new long[] { 2, 5 }, randomVector(10, 4))),
				OptionalFloat.empty());

		LoraAdapterSet set = GgufLoraImporter.importFrom(gguf, OptionalDouble.empty());
		assertThat(set.size()).isEqualTo(2);
		assertThat(set.get(0, "wq")).isNotNull();
		assertThat(set.get(1, "wdown")).isNotNull();
	}

	@Test
	@DisplayName("accepts the optional trailing .weight suffix on the lora tensor names")
	void accepts_trailing_weight_suffix(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_v.weight.lora_a.weight", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_v.weight.lora_b.weight", new long[] { 2, 3 }, randomVector(6, 2))),
				OptionalFloat.empty());

		LoraAdapterSet set = GgufLoraImporter.importFrom(gguf, OptionalDouble.empty());
		assertThat(set.get(0, "wv")).isNotNull();
	}

	@Test
	@DisplayName("explicit alpha override wins over GGUF metadata alpha")
	void alpha_override_wins(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_q.weight.lora_b", new long[] { 2, 3 }, randomVector(6, 2))),
				OptionalFloat.of(16f));

		LoraAdapterSet withMeta = GgufLoraImporter.importFrom(gguf, OptionalDouble.empty());
		assertThat(withMeta.get(0, "wq").scale).isCloseTo(16f / 2f, within(1e-6f));

		LoraAdapterSet withOverride = GgufLoraImporter.importFrom(gguf, OptionalDouble.of(4.0));
		assertThat(withOverride.get(0, "wq").scale).isCloseTo(4f / 2f, within(1e-6f));
	}

	@Test
	@DisplayName("unrecognized tensor name fails closed")
	void unrecognized_tensor_name_fails_closed(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_q.weight.lora_b", new long[] { 2, 3 }, randomVector(6, 2)),
						tensor("some_unrelated_tensor", new long[] { 4 }, randomVector(4, 9))),
				OptionalFloat.empty());

		assertThatThrownBy(() -> GgufLoraImporter.importFrom(gguf, OptionalDouble.empty()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Unrecognized tensor");
	}

	@Test
	@DisplayName("unsupported projection name fails closed")
	void unsupported_projection_fails_closed(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_weird.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_weird.weight.lora_b", new long[] { 2, 3 }, randomVector(6, 2))),
				OptionalFloat.empty());

		assertThatThrownBy(() -> GgufLoraImporter.importFrom(gguf, OptionalDouble.empty()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Unsupported LoRA projection");
	}

	@Test
	@DisplayName("missing lora_b half fails closed")
	void missing_half_fails_closed(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1))),
				OptionalFloat.empty());

		assertThatThrownBy(() -> GgufLoraImporter.importFrom(gguf, OptionalDouble.empty()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("missing its lora_b half");
	}

	@Test
	@DisplayName("mismatched rank between lora_a and lora_b fails closed")
	void rank_mismatch_fails_closed(@TempDir Path dir) throws IOException {
		Path gguf = writeFixture(dir, "adapter.gguf",
				List.of(tensor("blk.0.attn_q.weight.lora_a", new long[] { 4, 2 }, randomVector(8, 1)),
						tensor("blk.0.attn_q.weight.lora_b", new long[] { 3, 3 }, randomVector(9, 2))),
				OptionalFloat.empty());

		assertThatThrownBy(() -> GgufLoraImporter.importFrom(gguf, OptionalDouble.empty()))
				.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("rank mismatch");
	}

	// ── fixture GGUF writer ───────────────────────────────────────────────────

	private record FixtureTensor(String name, long[] dims, float[] data) {
	}

	private static FixtureTensor tensor(String name, long[] dims, float[] data) {
		return new FixtureTensor(name, dims, data);
	}

	/** Sentinel wrapper so a missing alpha is expressible without importing java.util.Optional twice. */
	private static final class OptionalFloat {
		private final Float value;

		private OptionalFloat(Float value) {
			this.value = value;
		}

		static OptionalFloat empty() {
			return new OptionalFloat(null);
		}

		static OptionalFloat of(float v) {
			return new OptionalFloat(v);
		}
	}

	private static final int GGML_TYPE_F32 = 0;
	private static final int GGUF_MAGIC = 0x46554747;
	private static final int GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6;
	private static final int ALIGNMENT = 32;

	private static Path writeFixture(Path dir, String filename, List<FixtureTensor> tensors, OptionalFloat metaAlpha)
			throws IOException {
		int kvCount = metaAlpha.value != null ? 1 : 0;

		int headerSize = 24;
		int kvSize = 0;
		if (metaAlpha.value != null) {
			byte[] keyBytes = "adapter.lora.alpha".getBytes(StandardCharsets.UTF_8);
			kvSize = 8 + keyBytes.length + 4 + 4; // keyLen+key + valueType + float
		}

		int infoSize = 0;
		for (FixtureTensor t : tensors) {
			byte[] nameBytes = t.name().getBytes(StandardCharsets.UTF_8);
			infoSize += 8 + nameBytes.length + 4 + 8L * t.dims().length + 4 + 8;
		}

		int prePad = headerSize + kvSize + infoSize;
		int aligned = ((prePad + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
		int padLen = aligned - prePad;

		long totalDataBytes = 0;
		long[] offsets = new long[tensors.size()];
		for (int i = 0; i < tensors.size(); i++) {
			offsets[i] = totalDataBytes;
			totalDataBytes += tensors.get(i).data().length * 4L;
		}

		ByteBuffer buf = ByteBuffer.allocate((int) (aligned + totalDataBytes)).order(ByteOrder.LITTLE_ENDIAN);

		buf.putInt(GGUF_MAGIC);
		buf.putInt(3);
		buf.putLong(tensors.size());
		buf.putLong(kvCount);

		if (metaAlpha.value != null) {
			byte[] keyBytes = "adapter.lora.alpha".getBytes(StandardCharsets.UTF_8);
			buf.putLong(keyBytes.length);
			buf.put(keyBytes);
			buf.putInt(GGUF_METADATA_VALUE_TYPE_FLOAT32);
			buf.putFloat(metaAlpha.value);
		}

		for (int i = 0; i < tensors.size(); i++) {
			FixtureTensor t = tensors.get(i);
			byte[] nameBytes = t.name().getBytes(StandardCharsets.UTF_8);
			buf.putLong(nameBytes.length);
			buf.put(nameBytes);
			buf.putInt(t.dims().length);
			for (long d : t.dims())
				buf.putLong(d);
			buf.putInt(GGML_TYPE_F32);
			buf.putLong(offsets[i]);
		}

		buf.put(new byte[padLen]);

		for (FixtureTensor t : tensors)
			for (float f : t.data())
				buf.putFloat(f);

		Path gguf = dir.resolve(filename);
		Files.write(gguf, buf.array());
		return gguf;
	}

	private static float[] randomVector(int n, long seed) {
		Random r = new Random(seed);
		float[] v = new float[n];
		for (int i = 0; i < n; i++)
			v[i] = (float) (r.nextGaussian() * 0.2);
		return v;
	}
}
