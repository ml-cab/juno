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

package cab.ml.juno.vision;

import java.nio.file.Path;

/**
 * Resolves which GGUF file holds the CLIP vision-encoder tensors for a given
 * text model.
 *
 * <p>Real-world llama.cpp-format multimodal releases (LLaVA, Qwen-VL,
 * SmolVLM, etc.) ship the vision encoder in a <b>separate</b> GGUF file,
 * conventionally named {@code mmproj-*.gguf}, loaded alongside the base LLM
 * via a {@code --mmproj} flag. The base LLM file itself never contains
 * {@code v.*} / {@code mm.*} tensors — those live only in the mmproj file.
 * A single file containing both the LLM and the CLIP encoder ("merged
 * format") is not how any known public GGUF is distributed; earlier Juno
 * code assumed such a format and consequently classified every real
 * downloaded I2T model as text-only.
 *
 * <p>This class centralizes the (trivial but previously-missing) decision of
 * which path to open when probing for, or loading, vision tensors:
 * <ul>
 *   <li>if an explicit mmproj path is supplied, vision tensors are read from
 *       it, never from the text model file;
 *   <li>otherwise the text model file itself is probed, so a genuinely
 *       merged single-file GGUF (should one ever exist) still works.
 * </ul>
 */
public record VisionModelPaths(Path textModelPath, Path visionWeightsPath) {

    public VisionModelPaths {
        if (textModelPath == null)
            throw new IllegalArgumentException("textModelPath must not be null");
        if (visionWeightsPath == null)
            throw new IllegalArgumentException("visionWeightsPath must not be null");
    }

    /**
     * @param modelPath  path to the base LLM GGUF (required)
     * @param mmprojPath path to a separate mmproj GGUF holding the CLIP vision
     *                   encoder, or {@code null} if none was given — in which
     *                   case {@code modelPath} itself is probed for vision
     *                   tensors (merged-file fallback)
     */
    public static VisionModelPaths of(Path modelPath, Path mmprojPath) {
        return new VisionModelPaths(modelPath, mmprojPath != null ? mmprojPath : modelPath);
    }

    /** True when vision tensors are read from a file other than the text model. */
    public boolean usesSeparateMmproj() {
        return !visionWeightsPath.equals(textModelPath);
    }
}