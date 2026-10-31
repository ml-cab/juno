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
package cab.ml.juno.kit;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import cab.ml.juno.player.LoraTrainer;

/**
 * A LoRA adapter for probes that need an overlay to play back: trained in the run, for one step on one
 * sentence, so nothing is downloaded. Its effect on the output is negligible; that it loads and is applied
 * is what those probes check.
 */
final class KitAdapters {

	static final String TEXT = "The kit trains this adapter for one step so the overlay path has a real file to load.";

	private KitAdapters() {
	}

	static Path trainSmall(Path model, Path adapter) throws IOException {
		Files.createDirectories(adapter.toAbsolutePath().getParent());
		try (LoraTrainer t = LoraTrainer.open(model, adapter, 4, 8f, 1e-4)) {
			t.trainRawText(TEXT, 1, 32);
			t.save();
		}
		return adapter;
	}
}
