/*
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

package cab.ml.juno.metrics;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Extracts one named recording into one named metrics file.
 *
 * <pre>
 * java -cp &lt;jar&gt; cab.ml.juno.metrics.JfrMetricsCli &lt;recording.jfr&gt; &lt;metrics.json&gt; [model-stem] [model-filename]
 * </pre>
 *
 * <p>{@link MetricsMain} covers the two cases the running engine needs: extract
 * every recording found in the working directory and map it against
 * {@code models.json}, or extract programmatically at shutdown. Neither serves a
 * caller that starts and stops a recording itself and wants the result in a
 * particular file: both write to a fixed relative path, so the working directory
 * becomes the only way to steer the output, and the directory-scanning form also
 * requires the model to be listed in {@code models.json}.
 *
 * <p>A measurement harness needs exactly that missing form. Warming an engine up
 * before the measured request means the requests that must not be measured run in
 * the same process as the one that must, so the recording has to be scoped to the
 * measured request rather than to the process lifetime — and then the harness, not
 * the engine, owns the recording and names where its metrics go.
 *
 * <p>An absent or empty recording fails here rather than producing a metrics file
 * of zeroes. Every metric this project publishes is written on every run whether
 * or not its event fired, so a zero is a real reading; a zero that actually means
 * "the recording was never written" would be read as a run with no collection
 * pauses and no tokens, which looks exactly like a clean fast one.
 *
 * @author Yevhen Soldatov
 */
public final class JfrMetricsCli {

	private static final String USAGE = "usage: JfrMetricsCli <recording.jfr> <metrics.json> [model-stem] [model-filename]";

	private JfrMetricsCli() {
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 2 || args.length > 4)
			throw new IllegalArgumentException(USAGE);

		Path jfrFile = Path.of(args[0]);
		Path output = Path.of(args[1]);
		String modelStem = args.length > 2 ? args[2] : stemFrom(jfrFile);
		String modelFilename = args.length > 3 ? args[3] : modelStem;

		run(jfrFile, output, modelStem, modelFilename);
		System.out.println("Wrote metrics to " + output);
	}

	/**
	 * Extracts {@code jfrFile} and writes the metrics JSON to {@code output},
	 * creating parent directories as needed.
	 *
	 * @param jfrFile       the recording to read
	 * @param output        the metrics JSON to write
	 * @param modelStem     model name recorded in the output
	 * @param modelFilename model filename recorded in the output
	 * @throws IOException if the recording is missing or empty, or the output
	 *                     cannot be written
	 */
	static void run(Path jfrFile, Path output, String modelStem, String modelFilename) throws Exception {
		if (!Files.isRegularFile(jfrFile))
			throw new IOException("recording not found: " + jfrFile);
		if (Files.size(jfrFile) == 0)
			throw new IOException("recording is empty, nothing to extract: " + jfrFile);

		ModelsConfig.ModelEntry entry = new ModelsConfig.ModelEntry(modelStem, modelFilename);
		MetricsSnapshot.ModelMetrics snapshot = JfrMetricsExtractor.extract(jfrFile, entry);

		Path parent = output.toAbsolutePath().getParent();
		if (parent != null)
			Files.createDirectories(parent);
		MetricsWriter.write(output, List.of(snapshot));
	}

	/**
	 * Model stem from a recording filename, accepting both the
	 * {@code juno-<stem>-<timestamp>.jfr} name the engine writes and a plain name a
	 * harness chose itself.
	 */
	private static String stemFrom(Path jfrFile) {
		String fileName = jfrFile.getFileName().toString();
		String stem = JfrModelMapper.modelStemFromJfr(fileName);
		if (stem != null)
			return stem;
		return fileName.endsWith(".jfr") ? fileName.substring(0, fileName.length() - 4) : fileName;
	}
}
