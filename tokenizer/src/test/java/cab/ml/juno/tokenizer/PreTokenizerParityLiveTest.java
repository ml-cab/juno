package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import cab.ml.juno.node.GgufReader;

/**
 * Token-ID parity for the pre-tokenizer split each model file declares, and
 * proof that a file declaring nothing is unchanged by the dispatch.
 *
 * <p>The expected IDs for the two files that declare a type were produced by a
 * second engine's tokenizer on the same strings, with its BOS suppressed; Juno
 * prepends BOS according to {@code tokenizer.ggml.add_bos_token}, so the arrays
 * for a file whose metadata sets that flag carry the BOS in front.
 *
 * <p>The strings are the divergence classes measured across a 33-line probe
 * corpus: runs of whitespace before a word (both types) and groups of digits
 * (the two types group them differently).
 */
@DisplayName("Pre-tokenizer parity — declared splits, and files that declare none")
class PreTokenizerParityLiveTest {

	private static Path model(String fileName) {
		Path p = Path.of("models/" + fileName);
		return p.toFile().exists() ? p : Path.of("../models/" + fileName);
	}

	private static final String QWEN2_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf";
	private static final String LLAMA_BPE_FILE = "Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile";
	private static final String NO_KEY_BPE_FILE = "phi-2.Q4_K_M.gguf";
	private static final String NO_KEY_SPM_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
	private static final String DEFAULT_SPM_FILE = "Phi-3.5-mini-instruct-Q4_K_M.gguf";

	private static boolean qwen2Present() {
		return model(QWEN2_FILE).toFile().exists();
	}

	private static boolean llamaBpePresent() {
		return model(LLAMA_BPE_FILE).toFile().exists();
	}

	private static boolean noKeyBpePresent() {
		return model(NO_KEY_BPE_FILE).toFile().exists();
	}

	private static boolean noKeySpmPresent() {
		return model(NO_KEY_SPM_FILE).toFile().exists();
	}

	private static boolean defaultSpmPresent() {
		return model(DEFAULT_SPM_FILE).toFile().exists();
	}

	private static int[] encode(String fileName, String text) throws Exception {
		try (GgufReader r = GgufReader.open(model(fileName))) {
			return GgufTokenizer.load(r).encode(text);
		}
	}

	// ── files that declare a type: the split has to reach the token IDs ───────

	@Test
	@EnabledIf("qwen2Present")
	@DisplayName("qwen2: a whitespace run splits so the word keeps its leading space")
	void qwen2_whitespaceRun_matchesReferenceIds() throws Exception {
		// Without the split this reads [64, 256, 65]: the two spaces merge into one
		// piece and "b" is left bare.
		assertThat(encode(QWEN2_FILE, "a  b")).containsExactly(64, 220, 293);
		assertThat(encode(QWEN2_FILE, "hello   world")).containsExactly(14990, 256, 1879);
		assertThat(encode(QWEN2_FILE, "multi  space   run    here"))
				.containsExactly(26268, 220, 3550, 256, 1598, 262, 1588);
		assertThat(encode(QWEN2_FILE, "  leading")).containsExactly(220, 6388);
	}

	@Test
	@EnabledIf("qwen2Present")
	@DisplayName("qwen2: the cases that already agreed keep agreeing")
	void qwen2_unchangedCases_stillMatchReferenceIds() throws Exception {
		assertThat(encode(QWEN2_FILE, "1234")).containsExactly(16, 17, 18, 19);
		assertThat(encode(QWEN2_FILE, "don't stop")).containsExactly(15007, 944, 2936);
		assertThat(encode(QWEN2_FILE, "Juno 2026 costs $12,345.67"))
				.containsExactly(41, 16311, 220, 17, 15, 17, 21, 7049, 400, 16, 17, 11, 18, 19, 20, 13, 21, 22);
		assertThat(encode(QWEN2_FILE, "trailing  ")).containsExactly(376, 14277, 256);
	}

	@Test
	@EnabledIf("llamaBpePresent")
	@DisplayName("llama-bpe: digits group in threes, not by whatever the merge table allows")
	void llamaBpe_digitGrouping_matchesReferenceIds() throws Exception {
		// 128000 is this file's BOS, which it asks for; the reference IDs after it
		// were taken with BOS suppressed.
		assertThat(encode(LLAMA_BPE_FILE, "3.14159265")).containsExactly(128000, 18, 13, 9335, 20128, 2397);
		assertThat(encode(LLAMA_BPE_FILE, "2026-09-25")).containsExactly(128000, 2366, 21, 12, 2545, 12, 914);
		assertThat(encode(LLAMA_BPE_FILE, "0x1F 0b1010"))
				.containsExactly(128000, 15, 87, 16, 37, 220, 15, 65, 4645, 15);
	}

	@Test
	@EnabledIf("llamaBpePresent")
	@DisplayName("llama-bpe: whitespace runs split the same way as qwen2")
	void llamaBpe_whitespaceRun_matchesReferenceIds() throws Exception {
		assertThat(encode(LLAMA_BPE_FILE, "a  b")).containsExactly(128000, 64, 220, 293);
		assertThat(encode(LLAMA_BPE_FILE, "multi  space   run    here"))
				.containsExactly(128000, 27364, 220, 3634, 256, 1629, 262, 1618);
	}

	// ── files that declare nothing: byte-for-byte what they produced before ───

	@Test
	@EnabledIf("noKeyBpePresent")
	@DisplayName("a BPE file with no declared type keeps the whole-run path")
	void noDeclaredType_bpe_unchanged() throws Exception {
		// phi-2 declares tokenizer.ggml.model=gpt2 and no tokenizer.ggml.pre. These
		// are the IDs it produced before the key was read at all.
		assertThat(encode(NO_KEY_BPE_FILE, "a  b")).containsExactly(64, 220, 275);
		assertThat(encode(NO_KEY_BPE_FILE, "hello   world")).containsExactly(31373, 220, 220, 995);
		assertThat(encode(NO_KEY_BPE_FILE, "3.14159265")).containsExactly(18, 13, 1415, 19707, 22980);
	}

	@Test
	@EnabledIf("noKeySpmPresent")
	@DisplayName("a SentencePiece file with no declared type is untouched")
	void noDeclaredType_sentencePiece_unchanged() throws Exception {
		assertThat(encode(NO_KEY_SPM_FILE, "a  b")).containsExactly(1, 263, 259, 29890);
		assertThat(encode(NO_KEY_SPM_FILE, "3.14159265"))
				.containsExactly(1, 29871, 29941, 29889, 29896, 29946, 29896, 29945, 29929, 29906, 29953, 29945);
	}

	@Test
	@EnabledIf("defaultSpmPresent")
	@DisplayName("a SentencePiece file declaring 'default' is untouched")
	void defaultType_sentencePiece_unchanged() throws Exception {
		// The declared value governs BPE vocabularies only. A SentencePiece file
		// does not take a pre-tokenizer split, so reading the key must not change
		// what this model tokenizes.
		assertThat(encode(DEFAULT_SPM_FILE, "a  b")).containsExactly(263, 29871, 289);
		assertThat(encode(DEFAULT_SPM_FILE, "multi  space   run    here"))
				.containsExactly(2473, 29871, 2913, 259, 1065, 1678, 1244);
	}
}
