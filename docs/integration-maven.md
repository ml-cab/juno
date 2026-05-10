# Maven integration

Artifacts publish under **`cab.ml`** with release version **`0.1.0`** on Maven Central (publisher verified for **ml.cab**). Browse coordinates after deploy:

[https://central.sonatype.com/search?q=g:cab.ml+juno](https://central.sonatype.com/search?q=g:cab.ml+juno)

Until artifacts appear on Central, build locally with `mvn clean package -DskipTests` and depend on modules from the reactor or install into `~/.m2`.

## Typical coordinates

The reactor publishes library jars (`api`, `registry`, `coordinator`, `node`, `lora`, `tokenizer`, `sampler`, `kvcache`, `health`, `metrics`) plus shaded runnable jars **`juno-player`**, **`juno-node`**, **`juno-master`**.

Minimal dependency for embedding libraries (adjust artifact IDs as needed):

```xml
<dependency>
  <groupId>cab.ml</groupId>
  <artifactId>coordinator</artifactId>
  <version>0.1.0</version>
</dependency>
```

End-user CLI flows normally run the **`juno-player`** shaded jar (classpath includes launcher entry points). Fat **`juno-node`** / **`juno-master`** jars match systemd/AWS layouts documented in [howto.md](howto.md).

Full commands and flags are not duplicated here; see [howto.md](howto.md).

## Publishing (maintainers)

Deploy to Maven Central via Sonatype Central Portal:

1. Configure `settings.xml` with server **`central`** (portal user token).
2. Run:

```bash
mvn clean deploy -Prelease-sign -Pcentral-publish -DskipTests -DskipITs
```

Signing requires GPG (`release-sign` profile). Omit `-Pcentral-publish` for local `install` only.
