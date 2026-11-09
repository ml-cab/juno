package cab.ml.juno.player;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads a {@code /train-file} text. The text is trained on token for token, and a file saved on Windows ends its
 * lines with CR LF; read as it is, the CR would become a token before every newline of the training data. CR LF is
 * therefore read as LF, so a text trains the same adapter whichever editor saved it. A CR that does not end a line
 * is kept.
 */
final class TrainingTextFile {

	private TrainingTextFile() {
	}

	/** The file's UTF-8 text with every CR LF read as LF. */
	static String read(Path path) throws IOException {
		return Files.readString(path).replace("\r\n", "\n");
	}
}
