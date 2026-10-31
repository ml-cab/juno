#!/usr/bin/env sh
# ─────────────────────────────────────────────────────────────────────────────
# juno-kit.sh — hardware validation kit launcher (juno-kit.bat on Windows)
#
# Builds Juno and the kit with Maven, detects what this machine can run, runs
# Juno's checks and readings, and packs the results into one zip to send back.
# Needs JDK 25 or later and Maven 3.9 or later; nothing else.
#
#   ./juno-kit.sh                 asks what to run
#   ./juno-kit.sh --suite quick   TinyLlama only
#   ./juno-kit.sh --help          every option
# ─────────────────────────────────────────────────────────────────────────────
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${JAVA_HOME:-}" ]; then JAVA="$JAVA_HOME/bin/java"; else JAVA=java; fi
exec "$JAVA" "-Djuno.kit.root=$DIR" "$DIR/juno-kit/src/main/java/cab/ml/juno/kit/KitBootstrap.java" "$@"
