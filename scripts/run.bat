@echo off
setlocal EnableDelayedExpansion

rem juno - Windows runtime launcher (no Maven required)
rem Uses pre-built shade jars from target/.
rem Build first: mvn clean package -DskipTests
rem Requires: JDK 21+   Mirrors: scripts/run.sh

rem Step up from scripts\ to project root
set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "DIR=%%~fI"

set "PLAYER_JAR=%DIR%\player\target\player.jar"
set "LIVE_JAR=%DIR%\integration\target\integration.jar"

rem UTF-8 codepage + ANSI VTP for colours
chcp 65001 >nul 2>&1
reg add "HKCU\Console" /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

call :find_java
if errorlevel 1 exit /b 1

call :check_java_version
if errorlevel 1 exit /b 1

set "JVM_BASE=--enable-preview --enable-native-access=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED -XX:+UseG1GC -XX:+AlwaysPreTouch -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8"

rem Dispatch: peek at %~1 without consuming it.
rem Only shift if it is a known subcommand, otherwise fall into cluster as-is.
if "%~1"=="" goto :usage
if /i "%~1"=="cluster" ( shift & goto :cluster )
if /i "%~1"=="local"   ( shift & goto :local )
if /i "%~1"=="test"    ( shift & goto :test )
goto :cluster

rem ============================================================================
rem  cluster
rem ============================================================================
:cluster

set "MODEL=%MODEL_PATH%"
if "%DTYPE%"==""       set "DTYPE=FLOAT16"
if "%MAX_TOKENS%"==""  set "MAX_TOKENS=200"
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.6"
if "%TOP_K%"==""       set "TOP_K=20"
if "%TOP_P%"==""       set "TOP_P=0.95"
if "%HEAP%"==""        set "HEAP=4g"
set "VERBOSE=false"
if "%PTYPE%"=="" set "PTYPE=pipeline"
set "JFR_DURATION_CLUSTER="

:cluster_parse
if "%~1"=="" goto :cluster_done
if /i "%~1"=="--model-path" ( set "MODEL=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--pType"      ( set "PTYPE=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--ptype"      ( set "PTYPE=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--dtype"      ( set "DTYPE=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--max-tokens" ( set "MAX_TOKENS=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--temperature"( set "TEMPERATURE=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--top-k"      ( set "TOP_K=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--top-p"      ( set "TOP_P=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--heap"       ( set "HEAP=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--jfr"        ( set "JFR_DURATION_CLUSTER=%~2" & shift & shift & goto :cluster_parse )
if /i "%~1"=="--float16" ( set "DTYPE=FLOAT16" & shift & goto :cluster_parse )
if /i "%~1"=="--fp16"    ( set "DTYPE=FLOAT16" & shift & goto :cluster_parse )
if /i "%~1"=="--float32" ( set "DTYPE=FLOAT32" & shift & goto :cluster_parse )
if /i "%~1"=="--int8"    ( set "DTYPE=INT8"    & shift & goto :cluster_parse )
if /i "%~1"=="--verbose" ( set "VERBOSE=true"  & shift & goto :cluster_parse )
if /i "%~1"=="-v"        ( set "VERBOSE=true"  & shift & goto :cluster_parse )
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat cluster --model-path PATH [flags]
  echo      or: set MODEL_PATH=PATH ^&^& run.bat cluster [flags]
  echo.
  echo   --pType pipeline^|tensor  parallelism type (default pipeline)
  echo     pipeline: contiguous layer blocks, serial activation flow
  echo     tensor:   weight-matrix slices, all nodes in parallel (AllReduce)
  echo               Constraint: numHeads %% 3 == 0
  echo   --dtype FLOAT32^|FLOAT16^|INT8  (default FLOAT16)
  echo   --float16 / --fp16 / --float32 / --int8
  echo   --max-tokens N    (default 200)
  echo   --temperature F   (default 0.6)
  echo   --top-k N         (default 20)
  echo   --top-p F         (default 0.95)
  echo   --heap SIZE       (default 4g)
  echo   --jfr DURATION    Java Flight Recording  e.g. 5m 30s 1h
  echo                     Records from start, writes juno-^<timestamp^>.jfr on exit
  echo   --verbose / -v
  goto :eof
)
if "%MODEL%"=="" if exist "%~1" ( set "MODEL=%~1" & shift & goto :cluster_parse )
echo [ERR] Unknown cluster flag: %~1
echo       Run: run.bat cluster --help
exit /b 1

:cluster_done
if "%MODEL%"=="" (
  echo [ERR] Model path is required.
  echo       Usage: run.bat cluster --model-path PATH
  exit /b 1
)
if not exist "%MODEL%" ( echo [ERR] Model not found: "%MODEL%" & exit /b 1 )
call :require_jar "%PLAYER_JAR%" "player"
if errorlevel 1 exit /b 1

echo [WARN] Starting 3-node cluster  (pType=%PTYPE%  dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  heap=%HEAP%)
if /i "%VERBOSE%"=="true" echo [WARN] Verbose mode ON
echo [WARN] Ctrl-C to stop all nodes and exit
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

set "JFR_FLAG_CLUSTER="
if not "%JFR_DURATION_CLUSTER%"=="" (
  for /f "tokens=2 delims==" %%T in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "DT=%%T"
  set "JFR_TS=!DT:~0,8!-!DT:~8,6!"
  set "JFR_FLAG_CLUSTER=-XX:StartFlightRecording=duration=%JFR_DURATION_CLUSTER%,filename=juno-!JFR_TS!.jfr,settings=profile,dumponexit=true"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_CLUSTER%  output=juno-!JFR_TS!.jfr
)

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" "-Djuno.node.heap=%HEAP%" %JFR_FLAG_CLUSTER% -jar "%PLAYER_JAR%" --model-path "%MODEL%" --pType "%PTYPE%" --dtype "%DTYPE%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K% --top-p %TOP_P% %VERBOSE_FLAG%
goto :eof

rem ============================================================================
rem  local
rem ============================================================================
:local

set "MODEL=%MODEL_PATH%"
if "%DTYPE%"==""       set "DTYPE=FLOAT16"
if "%MAX_TOKENS%"==""  set "MAX_TOKENS=200"
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.6"
if "%TOP_K%"==""       set "TOP_K=20"
if "%TOP_P%"==""       set "TOP_P=0.95"
if "%HEAP%"==""        set "HEAP=4g"
if "%NODES%"==""       set "NODES=3"
set "VERBOSE=false"
set "JFR_DURATION_LOCAL="

:local_parse
if "%~1"=="" goto :local_done
if /i "%~1"=="--model-path" ( set "MODEL=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--dtype"      ( set "DTYPE=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--max-tokens" ( set "MAX_TOKENS=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--temperature"( set "TEMPERATURE=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--top-k"      ( set "TOP_K=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--top-p"      ( set "TOP_P=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--heap"       ( set "HEAP=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--nodes"      ( set "NODES=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--jfr"        ( set "JFR_DURATION_LOCAL=%~2" & shift & shift & goto :local_parse )
if /i "%~1"=="--float16" ( set "DTYPE=FLOAT16" & shift & goto :local_parse )
if /i "%~1"=="--fp16"    ( set "DTYPE=FLOAT16" & shift & goto :local_parse )
if /i "%~1"=="--float32" ( set "DTYPE=FLOAT32" & shift & goto :local_parse )
if /i "%~1"=="--int8"    ( set "DTYPE=INT8"    & shift & goto :local_parse )
if /i "%~1"=="--verbose" ( set "VERBOSE=true"  & shift & goto :local_parse )
if /i "%~1"=="-v"        ( set "VERBOSE=true"  & shift & goto :local_parse )
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat local --model-path PATH [flags]
  echo      or: set MODEL_PATH=PATH ^&^& run.bat local [flags]
  echo.
  echo   --dtype FLOAT32^|FLOAT16^|INT8  (default FLOAT16)
  echo   --max-tokens N    (default 200)
  echo   --temperature F   (default 0.6)
  echo   --top-k N         (default 20)
  echo   --top-p F         (default 0.95)
  echo   --nodes N         (default 3)
  echo   --heap SIZE       (default 4g)
  echo   --jfr DURATION    Java Flight Recording  e.g. 5m 30s 1h
  echo                     Records from start, writes juno-^<timestamp^>.jfr on exit
  echo   --verbose / -v
  goto :eof
)
if "%MODEL%"=="" if exist "%~1" ( set "MODEL=%~1" & shift & goto :local_parse )
echo [ERR] Unknown local flag: %~1
echo       Run: run.bat local --help
exit /b 1

:local_done
if "%MODEL%"=="" (
  echo [ERR] Model path is required.
  echo       Usage: run.bat local --model-path PATH
  exit /b 1
)
if not exist "%MODEL%" ( echo [ERR] Model not found: "%MODEL%" & exit /b 1 )
call :require_jar "%PLAYER_JAR%" "player"
if errorlevel 1 exit /b 1

echo [INFO] Starting local in-process REPL  (dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  nodes=%NODES%  heap=%HEAP%)
if /i "%VERBOSE%"=="true" echo [WARN] Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

set "JFR_FLAG_LOCAL="
if not "%JFR_DURATION_LOCAL%"=="" (
  for /f "tokens=2 delims==" %%T in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "DT=%%T"
  set "JFR_TS=!DT:~0,8!-!DT:~8,6!"
  set "JFR_FLAG_LOCAL=-XX:StartFlightRecording=duration=%JFR_DURATION_LOCAL%,filename=juno-!JFR_TS!.jfr,settings=profile,dumponexit=true"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_LOCAL%  output=juno-!JFR_TS!.jfr
)

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" %JFR_FLAG_LOCAL% -jar "%PLAYER_JAR%" --model-path "%MODEL%" --dtype "%DTYPE%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K% --top-p %TOP_P% --nodes %NODES% --local %VERBOSE_FLAG%
goto :eof

rem ============================================================================
rem  test
rem ============================================================================
:test

set "MODEL=%MODEL_PATH%"
if "%HEAP%"==""  set "HEAP=4g"
if "%PTYPE%"=="" set "PTYPE=all"
set "JFR_DURATION_TEST="

:test_parse
if "%~1"=="" goto :test_done
if /i "%~1"=="--model-path" ( set "MODEL=%~2" & shift & shift & goto :test_parse )
if /i "%~1"=="--heap"       ( set "HEAP=%~2"  & shift & shift & goto :test_parse )
if /i "%~1"=="--pType"      ( set "PTYPE=%~2" & shift & shift & goto :test_parse )
if /i "%~1"=="--ptype"      ( set "PTYPE=%~2" & shift & shift & goto :test_parse )
if /i "%~1"=="--jfr"        ( set "JFR_DURATION_TEST=%~2" & shift & shift & goto :test_parse )
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat test --model-path PATH [flags]
  echo      or: run.bat test PATH
  echo.
  echo   Runs ModelLiveRunner - 8 real-model checks, exits 0 or 1.
  echo.
  echo   Pipeline-parallel (tests 1-6):
  echo     1. hello greeting coherence
  echo     2. no raw SentencePiece markers
  echo     3. question response is non-empty
  echo     4. greedy sampling is deterministic
  echo     5. multi-turn conversation accumulates context
  echo     6. FLOAT16 pipeline produces non-empty output
  echo   Tensor-parallel (tests 7-8):
  echo     7. tensor-parallel generation via AllReduce
  echo     8. tensor-parallel greedy determinism
  echo.
  echo   --pType pipeline^|tensor^|all   filter cluster tests (default: all)
  echo   --heap SIZE                    (default 4g)
  echo   --jfr DURATION                 Java Flight Recording  e.g. 5m 30s 1h
  echo                                  Records from start, writes juno-^<timestamp^>.jfr on exit
  goto :eof
)
if "%MODEL%"=="" if exist "%~1" ( set "MODEL=%~1" & shift & goto :test_parse )
echo [ERR] Unknown test flag: %~1
echo       Run: run.bat test --help
exit /b 1

:test_done
if "%MODEL%"=="" (
  echo [ERR] Model path is required.
  echo       Usage: run.bat test --model-path PATH
  exit /b 1
)
if not exist "%MODEL%" ( echo [ERR] Model not found: "%MODEL%" & exit /b 1 )
call :require_jar "%LIVE_JAR%" "integration"
if errorlevel 1 exit /b 1

echo [INFO] Running ModelLiveRunner  (pType=%PTYPE%  heap=%HEAP%)
echo.

set "JFR_FLAG_TEST="
if not "%JFR_DURATION_TEST%"=="" (
  for /f "tokens=2 delims==" %%T in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "DT=%%T"
  set "JFR_TS=!DT:~0,8!-!DT:~8,6!"
  set "JFR_FLAG_TEST=-XX:StartFlightRecording=duration=%JFR_DURATION_TEST%,filename=juno-!JFR_TS!.jfr,settings=profile,dumponexit=true"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_TEST%  output=juno-!JFR_TS!.jfr
)

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" "-DpType=%PTYPE%" "-Djuno.node.heap=%HEAP%" %JFR_FLAG_TEST% -jar "%LIVE_JAR%" "%MODEL%"
goto :eof

rem ============================================================================
rem  Usage
rem ============================================================================
:usage
echo.
echo juno runtime launcher  (Windows, no Maven - uses pre-built jars)
echo   Java:       %JAVA%
echo   player jar: %PLAYER_JAR%
echo   live jar:   %LIVE_JAR%
echo.
echo   Build first (one time):
echo     mvn clean package -DskipTests
echo.
echo   run.bat --model-path PATH            cluster (default)
echo   run.bat cluster --model-path PATH    3-node cluster + REPL
echo   run.bat local   --model-path PATH    in-process REPL (single JVM)
echo   run.bat test    --model-path PATH    6 smoke checks
echo.
echo   Env overrides: MODEL_PATH  DTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  NODES
echo   --jfr DURATION    Java Flight Recording  e.g. 5m 30s 1h  (all commands)
echo.
goto :eof

rem ============================================================================
rem  Helpers
rem ============================================================================

:find_java
if not "%JAVA_HOME%"=="" (
  if exist "%JAVA_HOME%\bin\java.exe" (
    set "JAVA=%JAVA_HOME%\bin\java.exe"
    exit /b 0
  )
)
for /f "usebackq tokens=* delims=" %%J in (`where java 2^>nul`) do (
  if not defined JAVA set "JAVA=%%J"
)
if not defined JAVA (
  echo [ERR] JDK not found. Install from https://adoptium.net and set JAVA_HOME.
  exit /b 1
)
exit /b 0

:check_java_version
set "JAVAVER_RAW="
for /f "usebackq tokens=3 delims= " %%V in (`"%JAVA%" -version 2^>^&1 ^| findstr /i "version"`) do (
  if not defined JAVAVER_RAW set "JAVAVER_RAW=%%V"
)
if not defined JAVAVER_RAW (
  echo [WARN] Unable to determine Java version. Continuing.
  exit /b 0
)
set "JAVAVER=%JAVAVER_RAW:"=%"
set "JAVAMAJOR="
for /f "tokens=1 delims=." %%M in ("%JAVAVER%") do set "JAVAMAJOR=%%M"
if "%JAVAMAJOR%"=="" ( echo [WARN] Unable to parse Java version. Continuing. & exit /b 0 )
if %JAVAMAJOR% LSS 21 (
  echo [ERR] JDK 21+ required (found: %JAVAVER%).
  exit /b 1
)
exit /b 0

:require_jar
if not exist %1 (
  echo [ERR] %~2 jar not found: %~1
  echo       Build first: mvn clean package -DskipTests
  exit /b 1
)
exit /b 0