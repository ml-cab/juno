package cab.ml.juno.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The pre-tokenizer split declared by {@code tokenizer.ggml.pre}.
 *
 * <p>Every expectation here is derived from the split each type specifies, and
 * the end-to-end token IDs those splits produce are pinned separately against a
 * second engine's tokenization in {@code PreTokenizerParityLiveTest}.
 */
@DisplayName("BpePreTokenizer — declared pre-tokenizer splits and the fail-closed set")
class BpePreTokenizerTest {

	private static List<String> qwen2(String text) {
		return BpePreTokenizer.resolve("qwen2").split(text);
	}

	private static List<String> llamaBpe(String text) {
		return BpePreTokenizer.resolve("llama-bpe").split(text);
	}

	// ── the split itself ──────────────────────────────────────────────────────

	@Test
	@DisplayName("a whitespace run keeps its last space with the following word")
	void whitespaceRun_lastSpaceJoinsNextWord() {
		// This is the divergence class that survives into token IDs: without the
		// split, a run of spaces merges into one vocabulary piece and the word
		// after it loses its leading space.
		assertThat(qwen2("a  b")).containsExactly("a", " ", " b");
		assertThat(qwen2("hello   world")).containsExactly("hello", "  ", " world");
		assertThat(llamaBpe("a  b")).containsExactly("a", " ", " b");
	}

	@Test
	@DisplayName("whitespace at end of input stays one run")
	void trailingWhitespace_staysOneRun() {
		assertThat(qwen2("trailing  ")).containsExactly("trailing", "  ");
	}

	@Test
	@DisplayName("qwen2 emits one piece per digit, llama-bpe groups digits in threes")
	void digitGrouping_differsBetweenTypes() {
		assertThat(qwen2("1234")).containsExactly("1", "2", "3", "4");
		assertThat(llamaBpe("1234")).containsExactly("123", "4");
		assertThat(llamaBpe("3.14159265")).containsExactly("3", ".", "141", "592", "65");
		assertThat(llamaBpe("2026-09-25")).containsExactly("202", "6", "-", "09", "-", "25");
	}

	@Test
	@DisplayName("contractions split at the apostrophe, in either case")
	void contractions_splitAtApostrophe() {
		assertThat(qwen2("don't stop")).containsExactly("don", "'t", " stop");
		assertThat(qwen2("I'LL go")).containsExactly("I", "'LL", " go");
	}

	@Test
	@DisplayName("a newline run is its own piece")
	void newlineRun_isOwnPiece() {
		assertThat(qwen2("a\n\nb")).containsExactly("a", "\n\n", "b");
		// A punctuation run absorbs the line break that follows it, so "end.\nNext"
		// is three pieces and not four.
		assertThat(qwen2("end.\nNext")).containsExactly("end", ".\n", "Next");
	}

	@Test
	@DisplayName("punctuation runs group together and carry at most one leading space")
	void punctuationRuns_groupWithOneLeadingSpace() {
		assertThat(qwen2("wait !!!")).containsExactly("wait", " !!!");
		assertThat(qwen2("{\"k\": 1}")).containsExactly("{\"", "k", "\":", " ", "1", "}");
	}

	@Test
	@DisplayName("every character of the input survives the split, for both types")
	void split_isLossless() {
		// A split that drops a character would silently shorten the prompt, which
		// no downstream assertion on token IDs would necessarily catch.
		String[] corpus = { "a  b", "hello   world", "  leading", "trailing  ", "3.14159265", "2026-09-25",
				"über straße", "日本語テスト", "emoji 🙂 test", "path/to/file.txt", "https://example.com/a?b=c",
				"{\"json\": [1, 2]}", "\ttabbed", "C++ code", "snake_case_name", "I'LL DO IT", "" };
		for (String text : corpus) {
			assertThat(String.join("", qwen2(text))).as("qwen2 coverage of %s", text).isEqualTo(text);
			assertThat(String.join("", llamaBpe(text))).as("llama-bpe coverage of %s", text).isEqualTo(text);
		}
	}

	// ── dispatch, and the fail-closed set ─────────────────────────────────────

	@Test
	@DisplayName("no declared type keeps the whole-run path")
	void resolve_absentOrDefault_returnsNull() {
		// null is the whole-segment merge every file used before the key was read;
		// the reference treats an absent key and 'default' as the same thing.
		assertThat(BpePreTokenizer.resolve(null)).isNull();
		assertThat(BpePreTokenizer.resolve("")).isNull();
		assertThat(BpePreTokenizer.resolve("  ")).isNull();
		assertThat(BpePreTokenizer.resolve("default")).isNull();
	}

	@Test
	@DisplayName("the implemented types resolve and name themselves")
	void resolve_implementedTypes() {
		assertThat(BpePreTokenizer.resolve("qwen2").typeName()).isEqualTo("qwen2");
		assertThat(BpePreTokenizer.resolve("llama-bpe").typeName()).isEqualTo("llama-bpe");
		assertThat(BpePreTokenizer.resolve("QWEN2").typeName()).as("declared value is matched case-insensitively")
				.isEqualTo("qwen2");
	}

	@Test
	@DisplayName("an unimplemented declared type is rejected by name")
	void resolve_unimplementedType_rejectedByName() {
		// Three real files on disk declare these. Running them under another
		// type's split would tokenize them differently from their training.
		for (String declared : new String[] { "tekken", "minimax-m2", "qwen35" }) {
			// A dedicated type, so a launcher can report it as a refused model file
			// rather than as a crash; it is still an IllegalArgumentException.
			assertThatThrownBy(() -> BpePreTokenizer.resolve(declared))
					.isInstanceOf(UnsupportedPreTokenizerException.class)
					.isInstanceOf(IllegalArgumentException.class)
					.hasMessageContaining(declared)
					.hasMessageContaining("tokenizer.ggml.pre");
		}
	}

	@Test
	@DisplayName("the rejection names what is implemented, so the reader can tell what to do")
	void rejection_listsImplementedTypes() {
		assertThatThrownBy(() -> BpePreTokenizer.resolve("gpt-4o"))
				.hasMessageContaining("qwen2")
				.hasMessageContaining("llama-bpe");
	}

	@Test
	@DisplayName("splitting is safe to call concurrently")
	void split_isThreadSafe() throws Exception {
		BpePreTokenizer pre = BpePreTokenizer.resolve("qwen2");
		assertThatCode(() -> {
			Thread[] threads = new Thread[4];
			for (int i = 0; i < threads.length; i++) {
				threads[i] = new Thread(() -> {
					for (int n = 0; n < 200; n++)
						assertThat(pre.split("a  b 1234")).containsExactly("a", " ", " b", " ", "1", "2", "3", "4");
				});
				threads[i].start();
			}
			for (Thread t : threads)
				t.join();
		}).doesNotThrowAnyException();
	}
}
