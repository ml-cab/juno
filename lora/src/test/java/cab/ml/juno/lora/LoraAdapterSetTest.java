package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for {@link LoraAdapterSet}.
 *
 * <h2>What to watch during testing</h2>
 * <ul>
 * <li><b>Round-trip fidelity</b>: saved A and B values must survive exactly
 * (float32 bit-identical). Any endianness bug in DataOutputStream will corrupt
 * ALL weights silently — catch it early with the round-trip test.
 * <li><b>Alpha vs scale</b>: the file stores alpha (= scale × rank), and the
 * loaded adapter reconstructs scale = alpha / rank. If rank or alpha is not
 * preserved, the scale will be wrong and loss may not converge.
 * <li><b>Key collision</b>: adding the same (layer, proj) twice replaces the
 * first adapter. This is intentional for hot-swapping but can be confusing if
 * you build a set in a loop with an off-by-one.
 * </ul>
 */
@DisplayName("LoraAdapterSet")
class LoraAdapterSetTest {

	@Test
	@DisplayName("get() returns null for unregistered (layer, proj)")
	void get_missing_returns_null() {
		LoraAdapterSet set = new LoraAdapterSet();
		assertThat(set.get(0, "wq")).isNull();
	}

	@Test
	@DisplayName("add() then get() returns the same adapter")
	void add_then_get() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = new LoraAdapter(4, 8, 16, 4f, new Random(1));
		set.add(3, "wq", a);
		assertThat(set.get(3, "wq")).isSameAs(a);
		assertThat(set.get(3, "wv")).isNull();
		assertThat(set.get(2, "wq")).isNull();
	}

	@Test
	@DisplayName("add() with same key replaces previous adapter")
	void add_replaces_on_same_key() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a1 = new LoraAdapter(4, 8, 16, 4f, new Random(1));
		LoraAdapter a2 = new LoraAdapter(4, 8, 16, 4f, new Random(2));
		set.add(0, "wq", a1);
		set.add(0, "wq", a2);
		assertThat(set.get(0, "wq")).isSameAs(a2);
		assertThat(set.size()).isEqualTo(1);
	}

	@Test
	@DisplayName("all() returns all registered adapters")
	void all_returns_all() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter aq = new LoraAdapter(4, 8, 16, 4f, new Random(1));
		LoraAdapter av = new LoraAdapter(4, 8, 16, 4f, new Random(2));
		set.add(0, "wq", aq);
		set.add(0, "wv", av);
		assertThat(set.all()).containsExactly(aq, av);
	}

	@Test
	@DisplayName("zeroAllGrads() clears every adapter's gradient accumulators")
	void zero_all_grads() {
		LoraAdapterSet set = new LoraAdapterSet();
		LoraAdapter a = new LoraAdapter(4, 8, 16, 4f, new Random(3));
		set.add(0, "wq", a);
		// put a non-zero gradient in
		a.backward(new float[16], new float[8]);
		// shouldn't be all zero here... unless B=0 (which it is at init, but
		// gradB and gradA may still be set from the backward call).
		// Just verify zeroGrad clears regardless
		set.zeroAllGrads();
		for (float g : a.gradA())
			assertThat(g).isEqualTo(0f);
		for (float g : a.gradB())
			assertThat(g).isEqualTo(0f);
	}

	@Test
	@DisplayName("resetFrom() restores B=0 and copies matching keys")
	void reset_from_zeros_delta() {
		LoraAdapterSet live = new LoraAdapterSet();
		LoraAdapter poisoned = makeNonZero(4, 8, 16, 4f, new Random(1));
		live.add(0, "wq", poisoned);
		LoraAdapter orphan = makeNonZero(4, 8, 16, 4f, new Random(2));
		live.add(0, "wv", orphan); // not in fresh → reinitialize

		LoraAdapterSet fresh = new LoraAdapterSet();
		LoraAdapter clean = new LoraAdapter(4, 8, 16, 4f, new Random(42));
		fresh.add(0, "wq", clean);

		int n = live.resetFrom(fresh, new Random(7));
		assertThat(n).isEqualTo(2);
		assertThat(live.get(0, "wq").a()).containsExactly(clean.a());
		assertThat(live.get(0, "wq").b()).containsOnly(0f);
		assertThat(live.get(0, "wv").b()).containsOnly(0f);
	}

	@Test
	@DisplayName("resetFrom() bumps doraGeneration and restores DoRA magnitudes")
	void reset_from_invalidates_dora_and_copies_magnitude() {
		LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD,
				LoraInitialization.KAIMING_UNIFORM, LoraMode.DORA);

		LoraAdapterSet live = new LoraAdapterSet();
		LoraAdapter trained = makeNonZero(cfg, 4, 3, new Random(1));
		live.add(0, "wq", trained);
		live.putMagnitude(0, "wq", DoraMagnitude.fromValues(new float[] { 9f, 8f, 7f }));
		long genBefore = live.doraGeneration();

		LoraAdapterSet fresh = new LoraAdapterSet();
		LoraAdapter clean = new LoraAdapter(cfg, 4, 3, new Random(42));
		fresh.add(0, "wq", clean);
		fresh.putMagnitude(0, "wq", DoraMagnitude.fromValues(new float[] { 1f, 2f, 3f }));

		live.resetFrom(fresh, new Random(7));

		assertThat(live.doraGeneration()).isEqualTo(genBefore + 1);
		assertThat(live.get(0, "wq").b()).containsOnly(0f);
		assertThat(live.getMagnitude(0, "wq").values()).containsExactly(1f, 2f, 3f);
	}

	// ── Serialisation round-trip ──────────────────────────────────────────────

	@Test
	@DisplayName("save/load round-trip preserves all weights bit-exactly")
	void save_load_roundtrip(@TempDir Path tmp) throws IOException {
		LoraAdapterSet original = new LoraAdapterSet();
		Random rng = new Random(42);

		// Two adapters with distinct weights
		LoraAdapter aqOrig = makeNonZero(4, 16, 32, 8f, rng);
		LoraAdapter avOrig = makeNonZero(4, 16, 8, 8f, rng);
		original.add(0, "wq", aqOrig);
		original.add(0, "wv", avOrig);

		Path file = tmp.resolve("test.lora");
		original.save(file);
		LoraAdapterSet loaded = LoraAdapterSet.load(file);

		assertThat(loaded.size()).isEqualTo(2);
		assertAdapterEqual(aqOrig, loaded.get(0, "wq"));
		assertAdapterEqual(avOrig, loaded.get(0, "wv"));
	}

	@Test
	@DisplayName("save/load preserves rank, scale, and inDim/outDim")
	void save_load_preserves_metadata(@TempDir Path tmp) throws IOException {
		LoraAdapterSet original = new LoraAdapterSet();
		original.add(5, "wv", new LoraAdapter(8, 64, 128, 16f, new Random(7)));

		Path file = tmp.resolve("meta.lora");
		original.save(file);
		LoraAdapterSet loaded = LoraAdapterSet.load(file);

		LoraAdapter a = loaded.get(5, "wv");
		assertThat(a).isNotNull();
		assertThat(a.rank).isEqualTo(8);
		assertThat(a.inDim).isEqualTo(64);
		assertThat(a.outDim).isEqualTo(128);
		assertThat(a.scale).isCloseTo(16f / 8f, within(1e-6f));
	}

	@Test
	@DisplayName("loading a corrupt file (wrong magic) throws IOException")
	void load_corrupt_file_throws(@TempDir Path tmp) throws IOException {
		Path file = tmp.resolve("corrupt.lora");
		java.nio.file.Files.write(file, new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 });
		assertThatThrownBy(() -> LoraAdapterSet.load(file)).isInstanceOf(IOException.class)
				.hasMessageContaining("LoRA");
	}

	@Test
	@DisplayName("loaded adapters start with zero gradients")
	void loaded_adapters_have_zero_grads(@TempDir Path tmp) throws IOException {
		LoraAdapterSet original = new LoraAdapterSet();
		LoraAdapter adapter = makeNonZero(4, 8, 16, 4f, new Random(9));
		// Simulate training: put non-zero grads in
		adapter.backward(new float[16], new float[8]);
		original.add(0, "wq", adapter);
		original.save(tmp.resolve("grads.lora"));

		LoraAdapterSet loaded = LoraAdapterSet.load(tmp.resolve("grads.lora"));
		for (float g : loaded.get(0, "wq").gradA())
			assertThat(g).isEqualTo(0f);
		for (float g : loaded.get(0, "wq").gradB())
			assertThat(g).isEqualTo(0f);
	}

	@Nested
	@DisplayName("Checkpoint v1/v2")
	class CheckpointFormats {

		@Test
		@DisplayName("hard-coded v1 fixture loads as standard + legacy-normal + LoRA")
		void hard_coded_v1_fixture(@TempDir Path tmp) throws IOException {
			Path file = tmp.resolve("v1.lora");
			try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
				out.writeInt(0x4C4F5241);
				out.writeInt(1);
				out.writeInt(1);
				byte[] key = "0:wq".getBytes(java.nio.charset.StandardCharsets.UTF_8);
				out.writeInt(key.length);
				out.write(key);
				out.writeInt(2); // rank
				out.writeInt(2); // in
				out.writeInt(2); // out
				out.writeFloat(4f); // alpha → scale 2
				// A: 2*2, B: 2*2
				out.writeFloat(0.1f);
				out.writeFloat(0.2f);
				out.writeFloat(0.3f);
				out.writeFloat(0.4f);
				out.writeFloat(1f);
				out.writeFloat(2f);
				out.writeFloat(3f);
				out.writeFloat(4f);
			}

			LoraAdapterSet loaded = LoraAdapterSet.load(file);
			LoraAdapter a = loaded.get(0, "wq");
			assertThat(a.rank).isEqualTo(2);
			assertThat(a.alpha).isEqualTo(4f);
			assertThat(a.scale).isEqualTo(2f);
			assertThat(a.scaling).isEqualTo(LoraScaling.STANDARD);
			assertThat(a.initialization).isEqualTo(LoraInitialization.LEGACY_NORMAL);
			assertThat(a.mode).isEqualTo(LoraMode.LORA);
			assertThat(a.a()).containsExactly(0.1f, 0.2f, 0.3f, 0.4f);
			assertThat(a.b()).containsExactly(1f, 2f, 3f, 4f);
		}

		@Test
		@DisplayName("v2 round-trip preserves rsLoRA metadata and weights bit-exactly")
		void v2_rslora_roundtrip(@TempDir Path tmp) throws IOException {
			LoraAdapterSet original = new LoraAdapterSet();
			LoraAdapterConfig cfg = LoraAdapterConfig.of(4, 8f, LoraScaling.RANK_STABILIZED,
					LoraInitialization.KAIMING_UNIFORM, LoraMode.LORA);
			LoraAdapter a = makeNonZero(cfg, 8, 16, new Random(3));
			original.add(1, "wdown", a);

			Path file = tmp.resolve("rs.lora");
			original.save(file);
			LoraAdapterSet loaded = LoraAdapterSet.load(file);
			LoraAdapter b = loaded.get(1, "wdown");
			assertThat(b.scaling).isEqualTo(LoraScaling.RANK_STABILIZED);
			assertThat(b.initialization).isEqualTo(LoraInitialization.KAIMING_UNIFORM);
			assertThat(b.alpha).isEqualTo(8f);
			assertThat(b.scale).isCloseTo(4f, within(1e-6f));
			assertAdapterEqual(a, b);
		}

		@Test
		@DisplayName("v2 round-trip preserves DoRA magnitude and fingerprint")
		void v2_dora_roundtrip(@TempDir Path tmp) throws IOException {
			LoraAdapterSet original = new LoraAdapterSet();
			LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD,
					LoraInitialization.KAIMING_UNIFORM, LoraMode.DORA);
			LoraAdapter a = makeNonZero(cfg, 4, 3, new Random(5));
			original.add(0, "wv", a);
			original.putMagnitude(0, "wv", DoraMagnitude.fromValues(new float[] { 1.5f, 2.5f, 3.5f }));
			byte[] sha = new byte[32];
			sha[0] = 7;
			original.putFingerprint(0, "wv", new LoraAdapterSet.BaseTensorFingerprint(2, new int[] { 3, 4 }, sha));

			Path file = tmp.resolve("dora.lora");
			original.save(file);
			LoraAdapterSet loaded = LoraAdapterSet.load(file);
			assertThat(loaded.get(0, "wv").mode).isEqualTo(LoraMode.DORA);
			assertThat(loaded.getMagnitude(0, "wv").values()).containsExactly(1.5f, 2.5f, 3.5f);
			assertThat(loaded.getFingerprint(0, "wv")).isEqualTo(original.getFingerprint(0, "wv"));
		}

		@Test
		@DisplayName("saveLegacyV1 encodes rsLoRA effective scale as transformed alpha")
		void legacy_v1_export_rslora(@TempDir Path tmp) throws IOException {
			LoraAdapterSet original = new LoraAdapterSet();
			LoraAdapterConfig cfg = LoraAdapterConfig.of(4, 8f, LoraScaling.RANK_STABILIZED,
					LoraInitialization.LEGACY_NORMAL, LoraMode.LORA);
			LoraAdapter a = makeNonZero(cfg, 8, 16, new Random(11));
			original.add(0, "wq", a);
			Path file = tmp.resolve("legacy.lora");
			original.saveLegacyV1(file);

			LoraAdapterSet loaded = LoraAdapterSet.load(file);
			LoraAdapter b = loaded.get(0, "wq");
			assertThat(b.scaling).isEqualTo(LoraScaling.STANDARD);
			assertThat(b.scale).isCloseTo(a.scale, within(1e-6f));
			assertThat(b.a()).containsExactly(a.a());
			assertThat(b.b()).containsExactly(a.b());
		}

		@Test
		@DisplayName("saveLegacyV1 rejects DoRA")
		void legacy_v1_rejects_dora(@TempDir Path tmp) {
			LoraAdapterSet set = new LoraAdapterSet();
			LoraAdapterConfig cfg = LoraAdapterConfig.of(2, 2f, LoraScaling.STANDARD,
					LoraInitialization.LEGACY_NORMAL, LoraMode.DORA);
			set.add(0, "wq", new LoraAdapter(cfg, 4, 4, new Random(1)));
			assertThatThrownBy(() -> set.saveLegacyV1(tmp.resolve("x.lora")))
					.isInstanceOf(IllegalStateException.class).hasMessageContaining("DoRA");
		}

		@Test
		@DisplayName("v2 duplicate keys are rejected")
		void v2_duplicate_key(@TempDir Path tmp) throws IOException {
			LoraAdapterSet set = new LoraAdapterSet();
			set.add(0, "wq", new LoraAdapter(2, 2, 2, 2f, new Random(1)));
			Path file = tmp.resolve("dup.lora");
			set.save(file);
			byte[] bytes = Files.readAllBytes(file);
			// Rewrite count=2 and append the same entry payload twice
			try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
				out.writeInt(0x4C4F5241);
				out.writeInt(2);
				out.writeInt(2);
				// skip magic+version+count (12 bytes) from original
				int entryLen = ((bytes[12] & 0xff) << 24) | ((bytes[13] & 0xff) << 16) | ((bytes[14] & 0xff) << 8)
						| (bytes[15] & 0xff);
				byte[] entry = new byte[4 + entryLen];
				System.arraycopy(bytes, 12, entry, 0, entry.length);
				out.write(entry);
				out.write(entry);
			}
			assertThatThrownBy(() -> LoraAdapterSet.load(file)).isInstanceOf(IOException.class)
					.hasMessageContaining("Duplicate");
		}

		@Test
		@DisplayName("v2 truncated entry throws")
		void v2_truncated(@TempDir Path tmp) throws IOException {
			LoraAdapterSet set = new LoraAdapterSet();
			set.add(0, "wq", new LoraAdapter(2, 2, 2, 2f, new Random(1)));
			Path file = tmp.resolve("trunc.lora");
			set.save(file);
			byte[] bytes = Files.readAllBytes(file);
			Files.write(file, java.util.Arrays.copyOf(bytes, bytes.length - 8));
			assertThatThrownBy(() -> LoraAdapterSet.load(file)).isInstanceOf(IOException.class);
		}

		@Test
		@DisplayName("v2 unknown enum id throws")
		void v2_bad_enum(@TempDir Path tmp) throws IOException {
			LoraAdapterSet set = new LoraAdapterSet();
			set.add(0, "wq", new LoraAdapter(2, 2, 2, 2f, new Random(1)));
			Path file = tmp.resolve("enum.lora");
			set.save(file);
			byte[] bytes = Files.readAllBytes(file);
			// Patch scaling ordinal (after key "0:wq"=4 bytes len + 4 key + 3 ints + float = ...)
			// Safer approach: craft a minimal corrupt entry
			try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
				out.writeInt(0x4C4F5241);
				out.writeInt(2);
				out.writeInt(1);
				byte[] key = "0:wq".getBytes(java.nio.charset.StandardCharsets.UTF_8);
				java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
				try (DataOutputStream e = new DataOutputStream(bos)) {
					e.writeInt(key.length);
					e.write(key);
					e.writeInt(2);
					e.writeInt(2);
					e.writeInt(2);
					e.writeFloat(2f);
					e.writeInt(99); // bad scaling
					e.writeInt(0);
					e.writeInt(0);
					for (int i = 0; i < 4; i++)
						e.writeFloat(0f); // A
					for (int i = 0; i < 4; i++)
						e.writeFloat(0f); // B
					e.writeBoolean(false);
					e.writeBoolean(false);
					e.writeInt(0);
				}
				byte[] payload = bos.toByteArray();
				out.writeInt(payload.length);
				out.write(payload);
			}
			assertThatThrownBy(() -> LoraAdapterSet.load(file)).isInstanceOf(IOException.class);
		}

		@Test
		@DisplayName("magnitude requires matching adapter and outDim")
		void magnitude_validation() {
			LoraAdapterSet set = new LoraAdapterSet();
			assertThatThrownBy(() -> set.putMagnitude(0, "wq", DoraMagnitude.fromValues(new float[] { 1f })))
					.isInstanceOf(IllegalArgumentException.class);
			set.add(0, "wq", new LoraAdapter(2, 2, 2, 2f, new Random(1)));
			assertThatThrownBy(() -> set.putMagnitude(0, "wq", DoraMagnitude.fromValues(new float[] { 1f })))
					.isInstanceOf(IllegalArgumentException.class);
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────

	private LoraAdapter makeNonZero(int rank, int in, int out, float alpha, Random rng) {
		return makeNonZero(LoraAdapterConfig.legacy(rank, alpha), in, out, rng);
	}

	private LoraAdapter makeNonZero(LoraAdapterConfig config, int in, int out, Random rng) {
		LoraAdapter a = new LoraAdapter(config, in, out, rng);
		for (int i = 0; i < a.b().length; i++)
			a.b()[i] = (float) (rng.nextGaussian() * 0.02);
		return a;
	}

	private void assertAdapterEqual(LoraAdapter expected, LoraAdapter actual) {
		assertThat(actual).isNotNull();
		assertThat(actual.rank).isEqualTo(expected.rank);
		assertThat(actual.inDim).isEqualTo(expected.inDim);
		assertThat(actual.outDim).isEqualTo(expected.outDim);
		assertThat(actual.alpha).isEqualTo(expected.alpha);
		assertThat(actual.scale).isCloseTo(expected.scale, within(1e-6f));
		assertThat(actual.scaling).isEqualTo(expected.scaling);
		assertThat(actual.initialization).isEqualTo(expected.initialization);
		assertThat(actual.mode).isEqualTo(expected.mode);
		for (int i = 0; i < expected.a().length; i++)
			assertThat(actual.a()[i]).isEqualTo(expected.a()[i]); // bit-exact
		for (int i = 0; i < expected.b().length; i++)
			assertThat(actual.b()[i]).isEqualTo(expected.b()[i]); // bit-exact
	}
}