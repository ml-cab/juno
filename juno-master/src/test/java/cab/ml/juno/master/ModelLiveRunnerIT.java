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
package cab.ml.juno.master;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import cab.ml.juno.master.ModelLiveChecks.LiveCheck;
import cab.ml.juno.master.ModelLiveChecks.Suite;
import cab.ml.juno.node.CpuMatVec;
import cab.ml.juno.node.ForwardPassHandlerLoader;
import cab.ml.juno.node.GgufReader;
import cab.ml.juno.node.LlamaConfig;
import cab.ml.juno.node.ShardContext;
import cab.ml.juno.player.ClusterHarness;

/**
 * Live model integration tests — run once per each model path listed in the
 * {@code MODELS} system property (comma-separated absolute GGUF paths).
 *
 * <p>Disabled by default. Activate with the {@code integration} Maven profile:
 *
 * <pre>
 *   mvn verify -pl juno-master -Pintegration \
 *       -DMODELS=/data/tinyllama.Q4_K_M.gguf,/data/phi-3.5.Q4_K_M.gguf
 * </pre>
 *
 * <p>The ten checks are {@link ModelLiveChecks}, the same ones {@code ./juno test}
 * runs through {@link ModelLiveRunner}; see that class for the list. Every check is
 * attempted even when an earlier one fails, and each failure is reported with its
 * reason. Forked cluster nodes size their heap from the model file unless
 * {@code -Djuno.node.heap} is set.
 *
 * <p>A model whose {@code general.architecture} has no verified handler is not run
 * through the checks: the test asserts instead that the loader rejects it with an
 * error naming the architecture (see {@code ForwardPassHandlerLoader}), and that a
 * pipeline- or tensor-parallel cluster started with it fails to start with that
 * reason.
 */
@DisplayName("Model Live Runner")
class ModelLiveRunnerIT {

    /**
     * Reads the {@code MODELS} system property (comma-separated GGUF paths).
     * Skips blank entries and paths that do not exist, logging a warning for each.
     * If no valid paths remain the stream is empty and JUnit reports no tests run.
     */
    static Stream<String> modelPaths() {
        String prop = System.getProperty("MODELS", "").strip();
        if (prop.isEmpty()) {
            System.err.println("[ModelLiveRunnerIT] MODELS property is not set — no tests will run.");
            return Stream.empty();
        }
        return Arrays.stream(prop.split(","))
                .map(String::strip)
                .filter(p -> {
                    if (p.isEmpty()) return false;
                    if (!Files.exists(Path.of(p))) {
                        System.err.println("[ModelLiveRunnerIT] Model not found, skipping: " + p);
                        return false;
                    }
                    return true;
                });
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("modelPaths")
    @DisplayName("Full suite")
    void testModel(String modelPath) throws Exception {
        String architecture = ModelLiveChecks.architecture(modelPath);
        if (!ForwardPassHandlerLoader.isSupportedArchitecture(architecture)) {
            assertUnsupportedArchitectureRejected(modelPath, architecture);
            return;
        }
        List<LiveCheck> results = ModelLiveChecks.run(modelPath, EnumSet.allOf(Suite.class), System.out);
        assertFalse(results.isEmpty(), "no check ran");
        assertAll("live checks for " + Path.of(modelPath).getFileName(),
                results.stream().map(r -> (Executable) () -> assertTrue(r.passed(),
                        "Test " + r.number() + " (" + r.name() + "): " + r.detail())));
    }

    /**
     * The loader must refuse an architecture it has no verified handler for, with an
     * error that names it, and a cluster started with such a model must fail to start
     * with the same reason instead of coming up with stub nodes. The in-process check
     * asserts the loader's own error (it runs before any tensor is read, so the shard
     * geometry passed there is irrelevant); the cluster checks assert that the failure
     * reaches the coordinator in both pipeline- and tensor-parallel mode.
     */
    private void assertUnsupportedArchitectureRejected(String modelPath, String architecture) throws Exception {
        ShardContext context = new ShardContext("n0", 0, 1, true, true, 8, 8, 2);
        IOException e = assertThrows(IOException.class,
                () -> ForwardPassHandlerLoader.load(Path.of(modelPath), context, CpuMatVec.INSTANCE));
        assertTrue(e.getMessage().contains("Unsupported model architecture")
                        && e.getMessage().contains("'" + architecture + "'"),
                "rejection must name the architecture '" + architecture + "' but was: " + e.getMessage());

        LlamaConfig cfg;
        try (GgufReader reader = GgufReader.open(Path.of(modelPath))) {
            cfg = LlamaConfig.from(reader);
        }
        assertClusterStartFails(ClusterHarness.threeNodes(modelPath, cfg.numLayers()), "pipeline", architecture);
        assertClusterStartFails(ClusterHarness.tensorNodes(modelPath, cfg.numLayers(), cfg.numHeads()), "tensor",
                architecture);
    }

    private void assertClusterStartFails(ClusterHarness harness, String mode, String architecture) throws Exception {
        try {
            RuntimeException e = assertThrows(RuntimeException.class, harness::start);
            assertTrue(e.getMessage().contains("Unsupported model architecture")
                            && e.getMessage().contains("'" + architecture + "'"),
                    mode + "-parallel cluster start must fail naming '" + architecture + "' but was: "
                            + e.getMessage());
        } finally {
            harness.stop();
        }
    }
}
