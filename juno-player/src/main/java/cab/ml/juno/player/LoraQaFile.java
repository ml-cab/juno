/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.player;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Loads a JSON array of Q&amp;A training pairs for {@code /train-file-qa}.
 *
 * <p>
 * Expected shape:
 *
 * <pre>{@code
 * [
 *   {"Q": "What is my name?", "A": "Dima"},
 *   {"Q": "Where do I live?", "A": "Kyiv"}
 * ]
 * }</pre>
 *
 * @author Yevhen Soldatov
 */
public final class LoraQaFile {

	private static final ObjectMapper JSON = new ObjectMapper();

	/** One question/answer fact from a QA training file. */
	public record Pair(String q, String a) {
	}

	private LoraQaFile() {
	}

	/**
	 * Read and validate a {@code .json} QA file. Root must be a non-empty array of
	 * objects with non-blank {@code Q} and {@code A} string fields.
	 *
	 * @param path path ending in {@code .json}
	 * @return immutable list of pairs in file order
	 * @throws IllegalArgumentException if the path, extension, or contents are invalid
	 * @throws IOException              if the file cannot be read
	 */
	public static List<Pair> load(Path path) throws IOException {
		if (path == null)
			throw new IllegalArgumentException("path is required");
		if (!Files.exists(path))
			throw new IllegalArgumentException("File not found: " + path);
		String name = path.getFileName() != null ? path.getFileName().toString() : path.toString();
		if (!name.toLowerCase(Locale.ROOT).endsWith(".json"))
			throw new IllegalArgumentException("QA training file must end with .json: " + name);
		try {
			return parse(Files.readString(path));
		} catch (IllegalArgumentException e) {
			throw new IllegalArgumentException(e.getMessage() + " (" + name + ")", e);
		}
	}

	/**
	 * Parse and validate a JSON array of {@code Q}/{@code A} objects (HTTP body or file contents).
	 *
	 * @param json UTF-8 JSON text
	 * @return immutable list of pairs in array order
	 * @throws IllegalArgumentException if the JSON is invalid or does not match the schema
	 */
	public static List<Pair> parse(String json) {
		if (json == null || json.isBlank())
			throw new IllegalArgumentException("QA JSON body is empty");
		JsonNode root;
		try {
			root = JSON.readTree(json);
		} catch (com.fasterxml.jackson.core.JsonProcessingException e) {
			throw new IllegalArgumentException("Invalid JSON: " + e.getOriginalMessage(), e);
		}
		if (root == null || !root.isArray())
			throw new IllegalArgumentException("QA training file root must be a JSON array");
		if (root.isEmpty())
			throw new IllegalArgumentException("QA training file array is empty");

		List<Pair> pairs = new ArrayList<>(root.size());
		for (int i = 0; i < root.size(); i++) {
			JsonNode el = root.get(i);
			if (el == null || !el.isObject())
				throw new IllegalArgumentException("QA pair at index " + i + " must be a JSON object");
			String q = textField(el, "Q", i);
			String a = textField(el, "A", i);
			pairs.add(new Pair(q, a));
		}
		return List.copyOf(pairs);
	}

	private static String textField(JsonNode obj, String key, int index) {
		JsonNode n = obj.get(key);
		if (n == null || n.isNull())
			throw new IllegalArgumentException("QA pair at index " + index + " missing \"" + key + "\"");
		if (!n.isTextual())
			throw new IllegalArgumentException(
					"QA pair at index " + index + " field \"" + key + "\" must be a string");
		String v = n.asText().strip();
		if (v.isEmpty())
			throw new IllegalArgumentException(
					"QA pair at index " + index + " field \"" + key + "\" must be non-blank");
		return v;
	}
}
