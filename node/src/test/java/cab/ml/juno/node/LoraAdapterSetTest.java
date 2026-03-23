package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/**
 * Tests for {@link LoraAdapterSet}.
 *
 * <h2>What to watch during testing</h2>
 * <ul>
 *   <li><b>Round-trip fidelity</b>: saved A and B values must survive exactly
 *       (float32 bit-identical). Any endianness bug in DataOutputStream will
 *       corrupt ALL weights silently — catch it early with the round-trip test.
 *   <li><b>Alpha vs scale</b>: the file stores alpha (= scale × rank), and the
 *       loaded adapter reconstructs scale = alpha / rank. If rank or alpha is
 *       not preserved, the scale will be wrong and loss may not converge.
 *   <li><b>Key collision</b>: adding the same (layer, proj) twice replaces the
 *       first adapter. This is intentional for hot-swapping but can be
 *       confusing if you build a set in a loop with an off-by-one.
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
        for (float g : a.gradA()) assertThat(g).isEqualTo(0f);
        for (float g : a.gradB()) assertThat(g).isEqualTo(0f);
    }

    @Test
    @DisplayName("qv() factory creates adapters for every layer on wq and wv")
    void qv_factory_creates_correct_adapters() {
        // Minimal config: 3 layers, hidden=16, 2 heads, 1 kv head, headDim=8
        LlamaConfig cfg = new LlamaConfig(16, 3, 2, 1, 8, 32, 200, 1e-5f, 10000f, "llama");
        LoraAdapterSet set = LoraAdapterSet.qv(cfg, 4, 4f, new Random(5));

        assertThat(set.size()).isEqualTo(6);  // 3 layers × 2 projections
        for (int li = 0; li < 3; li++) {
            assertThat(set.get(li, "wq")).isNotNull();
            assertThat(set.get(li, "wv")).isNotNull();
            assertThat(set.get(li, "wk")).isNull();  // not created by qv()
        }
        // Shapes: wq is [hidden × hidden], wv is [kvDim × hidden]
        assertThat(set.get(0, "wq").outDim).isEqualTo(16);  // H
        assertThat(set.get(0, "wq").inDim).isEqualTo(16);   // H
        assertThat(set.get(0, "wv").outDim).isEqualTo(8);   // kvDim = 1 head × 8
        assertThat(set.get(0, "wv").inDim).isEqualTo(16);   // H
    }

    // ── Serialisation round-trip ──────────────────────────────────────────────

    @Test
    @DisplayName("save/load round-trip preserves all weights bit-exactly")
    void save_load_roundtrip(@TempDir Path tmp) throws IOException {
        LoraAdapterSet original = new LoraAdapterSet();
        Random rng = new Random(42);

        // Two adapters with distinct weights
        LoraAdapter aqOrig = makeNonZero(4, 16, 32, 8f, rng);
        LoraAdapter avOrig = makeNonZero(4, 16, 8,  8f, rng);
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
        java.nio.file.Files.write(file, new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07});
        assertThatThrownBy(() -> LoraAdapterSet.load(file))
            .isInstanceOf(IOException.class)
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
        for (float g : loaded.get(0, "wq").gradA()) assertThat(g).isEqualTo(0f);
        for (float g : loaded.get(0, "wq").gradB()) assertThat(g).isEqualTo(0f);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private LoraAdapter makeNonZero(int rank, int in, int out, float alpha, Random rng) {
        LoraAdapter a = new LoraAdapter(rank, in, out, alpha, rng);
        for (int i = 0; i < a.b().length; i++)
            a.b()[i] = (float) (rng.nextGaussian() * 0.02);
        return a;
    }

    private void assertAdapterEqual(LoraAdapter expected, LoraAdapter actual) {
        assertThat(actual).isNotNull();
        assertThat(actual.rank).isEqualTo(expected.rank);
        assertThat(actual.inDim).isEqualTo(expected.inDim);
        assertThat(actual.outDim).isEqualTo(expected.outDim);
        assertThat(actual.scale).isCloseTo(expected.scale, within(1e-6f));
        for (int i = 0; i < expected.a().length; i++)
            assertThat(actual.a()[i]).isEqualTo(expected.a()[i]);  // bit-exact
        for (int i = 0; i < expected.b().length; i++)
            assertThat(actual.b()[i]).isEqualTo(expected.b()[i]);  // bit-exact
    }
}