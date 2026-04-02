// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class ProductivityMain {

    public static void main(String[] args) throws Exception {
        Path projectRoot = Path.of(System.getProperty("user.dir"));
        ModelsConfig models = loadModelsConfig(projectRoot);

        List<Path> jfrFiles = listJfrFiles(projectRoot);
        Map<Path, ModelsConfig.ModelEntry> mapping = JfrModelMapper.mapByModelStem(jfrFiles, models);

        if (mapping.isEmpty()) {
            System.out.println("No JFR files mapped to models (expected: juno-<modelStem>-YYYYMMDD-HHMMSS.jfr).");
            return;
        }

        List<MetricsSnapshot.ModelMetrics> snapshots = new ArrayList<>();
        for (Map.Entry<Path, ModelsConfig.ModelEntry> e : mapping.entrySet()) {
            snapshots.add(JfrMetricsExtractor.extract(e.getKey(), e.getValue()));
        }

        warnIfClusterCoordinatorRecording(snapshots);

        Path output = projectRoot.resolve("target").resolve("productivity").resolve("metrics.json");
        MetricsWriter.write(output, snapshots);

        System.out.println("Wrote metrics to " + projectRoot.relativize(output));
    }

    private static ModelsConfig loadModelsConfig(Path projectRoot) throws IOException {
        Path[] candidates = new Path[] {
                projectRoot.resolve("productivity/src/main/resources/models.json"),
                projectRoot.resolve("src/main/resources/models.json")
        };
        for (Path path : candidates) {
            if (Files.isRegularFile(path)) {
                return new ModelsConfigLoader().load(path);
            }
        }
        throw new IOException(
                "models.json not found. Tried productivity/src/main/resources/models.json and src/main/resources/models.json under "
                        + projectRoot);
    }

    private static List<Path> listJfrFiles(Path root) throws IOException {
        List<Path> result = new ArrayList<>();
        try (var stream = Files.list(root)) {
            stream.filter(p -> p.getFileName().toString().endsWith(".jfr")).forEach(result::add);
        }
        return result;
    }

    /**
     * In default cluster mode, {@code --jfr} attaches only to the coordinator JVM. MatVec and
     * ForwardPass fire in forked node processes, so those events are missing and many metrics stay 0.
     */
    private static void warnIfClusterCoordinatorRecording(List<MetricsSnapshot.ModelMetrics> snapshots) {
        for (MetricsSnapshot.ModelMetrics s : snapshots) {
            double matVec = s.getMetrics().getOrDefault("juno.MatVec.count", 0.0);
            double forward = s.getMetrics().getOrDefault("juno.ForwardPass.count", 0.0);
            if (matVec == 0.0 && forward == 0.0) {
                System.err.println();
                System.err.println("NOTE: Recording \"" + s.getJfrFileName()
                        + "\" has no juno.MatVec / juno.ForwardPass events.");
                System.err.println(
                        "      With ./juno (cluster), JFR runs on the coordinator only; inference runs in separate node JVMs.");
                System.err.println(
                        "      For full productivity metrics use: ./juno local --model-path ... --jfr <duration>");
                System.err.println(
                        "      (single JVM — all custom juno.* events appear in one .jfr file.)");
                System.err.println();
            }
        }
    }
}
