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
package cab.ml.juno.node;

import java.io.IOException;
import java.nio.file.Path;

/**
 * The first check every entry point runs on a model file, before it reads the config or the
 * tokenizer. A file whose architecture has no verified handler usually fails other checks too
 * (the files on hand also declare pre-tokenizer types Juno does not implement), and whichever
 * check runs first decides what the user is told. The architecture is the one that matters: a
 * file refused for its pre-tokenizer would still not load once that was fixed.
 */
public final class ModelFileGate {

	private ModelFileGate() {
	}

	/**
	 * @throws UnsupportedModelException naming the architecture when no verified handler exists
	 * @throws IOException               when the file cannot be read
	 */
	public static void requireLoadable(Path modelPath) throws IOException {
		String arch = ForwardPassHandlerLoader.readArchitecture(modelPath);
		if (!ForwardPassHandlerLoader.isSupportedArchitecture(arch))
			LlamaFamilyArchitectures.requireVerified(arch, modelPath);
	}
}
