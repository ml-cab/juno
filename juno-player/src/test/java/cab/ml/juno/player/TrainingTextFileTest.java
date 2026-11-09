package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A {@code /train-file} text is trained on token for token. A file saved on Windows ends its lines with CR LF, and
 * the CR would become a token before every newline of the training data, so the same text would train a different
 * adapter depending on the editor that saved it.
 */
@DisplayName("TrainingTextFile")
class TrainingTextFileTest {

	@TempDir
	Path dir;

	@Test
	@DisplayName("CR LF line ends are read as LF")
	void crlf_is_read_as_lf() throws Exception {
		Path p = Files.writeString(dir.resolve("notes.txt"), "first line\r\nsecond line\r\n\r\nlast");
		assertThat(TrainingTextFile.read(p)).isEqualTo("first line\nsecond line\n\nlast");
	}

	@Test
	@DisplayName("an LF file is read unchanged")
	void lf_is_unchanged() throws Exception {
		Path p = Files.writeString(dir.resolve("notes.txt"), "first line\nsecond line\n");
		assertThat(TrainingTextFile.read(p)).isEqualTo("first line\nsecond line\n");
	}

	@Test
	@DisplayName("a CR that does not end a line is kept")
	void a_lone_cr_is_kept() throws Exception {
		Path p = Files.writeString(dir.resolve("notes.txt"), "a\rb\r\nc");
		assertThat(TrainingTextFile.read(p)).isEqualTo("a\rb\nc");
	}
}
