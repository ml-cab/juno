package cab.ml.juno.lora;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.lora.LoraPlaySpec.Entry;

@DisplayName("LoraPlaySpec")
class LoraPlaySpecTest {

	@Test
	@DisplayName("bare path defaults to scale 1.0 (back-compat)")
	void bare_path_defaults_to_scale_one() {
		List<Entry> entries = LoraPlaySpec.parse("adapters/a.lora");
		assertThat(entries).containsExactly(new Entry(Path.of("adapters/a.lora"), 1.0f));
	}

	@Test
	@DisplayName("path:scale parses the trailing scale")
	void path_with_scale() {
		List<Entry> entries = LoraPlaySpec.parse("a.lora:0.5");
		assertThat(entries).containsExactly(new Entry(Path.of("a.lora"), 0.5f));
	}

	@Test
	@DisplayName("comma-separated multi-entry syntax")
	void multi_entry() {
		List<Entry> entries = LoraPlaySpec.parse("a.lora:0.5,b.lora:1.0,c.lora");
		assertThat(entries).containsExactly(new Entry(Path.of("a.lora"), 0.5f), new Entry(Path.of("b.lora"), 1.0f),
				new Entry(Path.of("c.lora"), 1.0f));
	}

	@Test
	@DisplayName("Windows drive-letter path without a scale is not misread as path 'C'")
	void windows_drive_letter_without_scale() {
		List<Entry> entries = LoraPlaySpec.parse("C:\\models\\a.lora");
		assertThat(entries).containsExactly(new Entry(Path.of("C:\\models\\a.lora"), 1.0f));
	}

	@Test
	@DisplayName("Windows drive-letter path with a trailing scale finds the right colon")
	void windows_drive_letter_with_scale() {
		List<Entry> entries = LoraPlaySpec.parse("C:\\models\\a.lora:0.5");
		assertThat(entries).containsExactly(new Entry(Path.of("C:\\models\\a.lora"), 0.5f));
	}

	@Test
	@DisplayName("whitespace around entries is stripped")
	void whitespace_stripped() {
		List<Entry> entries = LoraPlaySpec.parse(" a.lora:0.5 , b.lora ");
		assertThat(entries).containsExactly(new Entry(Path.of("a.lora"), 0.5f), new Entry(Path.of("b.lora"), 1.0f));
	}

	@Test
	@DisplayName("blank spec fails closed")
	void blank_spec_fails_closed() {
		assertThatThrownBy(() -> LoraPlaySpec.parse("")).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> LoraPlaySpec.parse(null)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	@DisplayName("empty entry between commas fails closed")
	void empty_entry_fails_closed() {
		assertThatThrownBy(() -> LoraPlaySpec.parse("a.lora,,b.lora")).isInstanceOf(IllegalArgumentException.class);
	}
}
