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
if /i "%~1"=="lora"    ( shift & goto :lora )
if /i "%~1"=="test"    ( shift & goto :test )
goto :cluster

rem ============================================================================
rem  Helper: prepend CUDA bin to PATH if GPU mode and CUDA available
rem ============================================================================
:prepend_cuda_path
if /i "%USE_GPU%"=="false" exit /b 0
if not "%CUDA_PATH%"=="" if exist "%CUDA_PATH%\bin" (
  set "PATH=%CUDA_PATH%\bin;%PATH%"
  exit /b 0
)
if not "%CUDA_HOME%"=="" if exist "%CUDA_HOME%\bin" (
  set "PATH=%CUDA_HOME%\bin;%PATH%"
  exit /b 0
)
exit /b 0

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
set "USE_GPU=true"
if not "%USE_GPU_ENV%"=="" (
  if /i "%USE_GPU_ENV%"=="false" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="0" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="no" set "USE_GPU=false"
)

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
if /i "%~1"=="--gpu"     ( set "USE_GPU=true"  & shift & goto :cluster_parse )
if /i "%~1"=="--cpu"     ( set "USE_GPU=false" & shift & goto :cluster_parse )
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
  echo   --gpu             use GPU when available (default)
  echo   --cpu             use CPU only
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

echo [WARN] Starting 3-node cluster  (pType=%PTYPE%  dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  heap=%HEAP%  gpu=%USE_GPU%)
if /i "%VERBOSE%"=="true" echo [WARN] Verbose mode ON
echo [WARN] Ctrl-C to stop all nodes and exit
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

set "GPU_FLAG=--gpu"
if /i "%USE_GPU%"=="false" set "GPU_FLAG=--cpu"

rem In cluster mode ConsoleMain manages JFR programmatically (startClusterJfr),
rem exactly as in local mode.  Pass --jfr as an app arg, not a JVM flag.
set "JFR_ARG_CLUSTER="
if not "%JFR_DURATION_CLUSTER%"=="" (
  set "JFR_ARG_CLUSTER=--jfr %JFR_DURATION_CLUSTER%"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_CLUSTER%  (programmatic recording, metrics auto-printed on exit)
)

call :prepend_cuda_path

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" "-Djuno.node.heap=%HEAP%" -jar "%PLAYER_JAR%" --model-path "%MODEL%" --pType "%PTYPE%" --dtype "%DTYPE%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K% --top-p %TOP_P% %GPU_FLAG% %JFR_ARG_CLUSTER% %VERBOSE_FLAG%
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
set "USE_GPU=true"
if not "%USE_GPU_ENV%"=="" (
  if /i "%USE_GPU_ENV%"=="false" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="0" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="no" set "USE_GPU=false"
)

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
if /i "%~1"=="--gpu"     ( set "USE_GPU=true"  & shift & goto :local_parse )
if /i "%~1"=="--cpu"     ( set "USE_GPU=false" & shift & goto :local_parse )
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
  echo   --gpu             use GPU when available (default)
  echo   --cpu             use CPU only
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

echo [INFO] Starting local in-process REPL  (dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  nodes=%NODES%  heap=%HEAP%  gpu=%USE_GPU%)
if /i "%VERBOSE%"=="true" echo [WARN] Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

set "GPU_FLAG=--gpu"
if /i "%USE_GPU%"=="false" set "GPU_FLAG=--cpu"

rem Local mode: pass --jfr as app arg so ConsoleMain.startLocalJfr() owns the
rem recording lifecycle and auto-prints metrics on exit (mirrors run.sh local).
set "JFR_ARG_LOCAL="
if not "%JFR_DURATION_LOCAL%"=="" (
  set "JFR_ARG_LOCAL=--jfr %JFR_DURATION_LOCAL%"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_LOCAL%  (programmatic recording, metrics auto-printed on exit)
)

call :prepend_cuda_path

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" -jar "%PLAYER_JAR%" --model-path "%MODEL%" --dtype "%DTYPE%" --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K% --top-p %TOP_P% --nodes %NODES% --local %GPU_FLAG% %JFR_ARG_LOCAL% %VERBOSE_FLAG%
goto :eof

rem ============================================================================
rem  lora
rem ============================================================================
:lora

set "MODEL=%MODEL_PATH%"
set "LORA_PATH_VAL=%LORA_PATH%"
if "%LORA_RANK%"==""  set "LORA_RANK=8"
if "%LORA_LR%"==""    set "LORA_LR=0.0001"
if "%LORA_STEPS%"=="" set "LORA_STEPS=50"
if "%MAX_TOKENS%"==""  set "MAX_TOKENS=200"
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.6"
if "%TOP_K%"==""       set "TOP_K=20"
if "%TOP_P%"==""       set "TOP_P=0.95"
if "%HEAP%"==""        set "HEAP=4g"
set "VERBOSE=false"
set "JFR_DURATION_LORA="
set "USE_GPU=true"
if not "%USE_GPU_ENV%"=="" (
  if /i "%USE_GPU_ENV%"=="false" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="0" set "USE_GPU=false"
  if /i "%USE_GPU_ENV%"=="no" set "USE_GPU=false"
)

:lora_parse
if "%~1"=="" goto :lora_done
if /i "%~1"=="--model-path"  ( set "MODEL=%~2"         & shift & shift & goto :lora_parse )
if /i "%~1"=="--lora-path"   ( set "LORA_PATH_VAL=%~2" & shift & shift & goto :lora_parse )
if /i "%~1"=="--lora-rank"   ( set "LORA_RANK=%~2"     & shift & shift & goto :lora_parse )
if /i "%~1"=="--lora-alpha"  ( set "LORA_ALPHA=%~2"    & shift & shift & goto :lora_parse )
if /i "%~1"=="--lora-lr"     ( set "LORA_LR=%~2"       & shift & shift & goto :lora_parse )
if /i "%~1"=="--lora-steps"  ( set "LORA_STEPS=%~2"    & shift & shift & goto :lora_parse )
if /i "%~1"=="--max-tokens"  ( set "MAX_TOKENS=%~2"    & shift & shift & goto :lora_parse )
if /i "%~1"=="--temperature" ( set "TEMPERATURE=%~2"   & shift & shift & goto :lora_parse )
if /i "%~1"=="--top-k"       ( set "TOP_K=%~2"         & shift & shift & goto :lora_parse )
if /i "%~1"=="--top-p"       ( set "TOP_P=%~2"         & shift & shift & goto :lora_parse )
if /i "%~1"=="--heap"        ( set "HEAP=%~2"           & shift & shift & goto :lora_parse )
if /i "%~1"=="--jfr"         ( set "JFR_DURATION_LORA=%~2" & shift & shift & goto :lora_parse )
if /i "%~1"=="--pType"       ( shift & shift & goto :lora_parse )
if /i "%~1"=="--ptype"       ( shift & shift & goto :lora_parse )
if /i "%~1"=="--verbose" ( set "VERBOSE=true" & shift & goto :lora_parse )
if /i "%~1"=="-v"        ( set "VERBOSE=true" & shift & goto :lora_parse )
if /i "%~1"=="--gpu"     ( set "USE_GPU=true"  & shift & goto :lora_parse )
if /i "%~1"=="--cpu"     ( set "USE_GPU=false" & shift & goto :lora_parse )
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat lora --model-path PATH [flags]
  echo      or: set MODEL_PATH=PATH ^&^& run.bat lora [flags]
  echo.
  echo   Runs a LoRA fine-tuning REPL in a single in-process JVM.
  echo   Adapter weights are saved to a separate .lora file.
  echo   The base GGUF is never modified.
  echo.
  echo   Required:
  echo     --model-path PATH       Path to a GGUF model file
  echo.
  echo   LoRA adapter:
  echo     --lora-path PATH        Checkpoint file  (default: ^<model^>.lora)
  echo     --lora-rank N           Low-rank dimension  (default: 8)
  echo     --lora-alpha F          Scaling alpha  (default = rank)
  echo     --lora-lr F             Adam learning rate  (default: 1e-4)
  echo     --lora-steps N          Gradient steps per /train  (default: 50)
  echo.
  echo   Generation (used for inference):
  echo     --max-tokens N          (default 200)
  echo     --temperature F         (default 0.6)
  echo     --top-k N               (default 20)
  echo     --top-p F               (default 0.95)
  echo.
  echo   JVM:
  echo     --heap SIZE             e.g. 4g 8g 16g  (default 4g)
  echo                             Use at least 2x the model file size.
  echo.
  echo   Backend:
  echo     --gpu                   use GPU when available (default)
  echo     --cpu                   use CPU only
  echo.
  echo   REPL commands:
  echo     /train ^<text^>          Fine-tune on inline text
  echo     /train-file ^<path^>     Fine-tune on a text file
  echo     /save                   Save adapter checkpoint
  echo     /reset                  Reinitialise adapters
  echo     /status                 Show adapter info
  echo     /merge-hint             Explain offline weight merge
  echo     Regular input           Chat with adapter applied
  echo.
  echo   Env overrides: MODEL_PATH LORA_PATH LORA_RANK LORA_ALPHA LORA_LR LORA_STEPS
  echo                  MAX_TOKENS TEMPERATURE TOP_K TOP_P HEAP USE_GPU
  echo.
  echo   Examples:
  echo     run.bat lora --model-path C:\models\tinyllama.gguf
  echo     run.bat lora --model-path C:\models\tinyllama.gguf --lora-rank 16 --heap 8g
  echo     run.bat lora --model-path C:\models\tinyllama.gguf --lora-path my.lora
  echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat lora
  goto :eof
)
if "%MODEL%"=="" if exist "%~1" ( set "MODEL=%~1" & shift & goto :lora_parse )
echo [ERR] Unknown lora flag: %~1
echo       Run: run.bat lora --help
exit /b 1

:lora_done
if "%MODEL%"=="" (
  echo [ERR] Model path is required.
  echo       Usage: run.bat lora --model-path PATH
  exit /b 1
)
if not exist "%MODEL%" ( echo [ERR] Model not found: "%MODEL%" & exit /b 1 )
call :require_jar "%PLAYER_JAR%" "player"
if errorlevel 1 exit /b 1

rem Default alpha = rank when not set
if "%LORA_ALPHA%"=="" set "LORA_ALPHA=%LORA_RANK%"

echo [INFO] Starting LoRA fine-tuning REPL  (rank=%LORA_RANK%  alpha=%LORA_ALPHA%  lr=%LORA_LR%  steps=%LORA_STEPS%  heap=%HEAP%  gpu=%USE_GPU%)
if not "%LORA_PATH_VAL%"=="" echo [INFO] Adapter file: %LORA_PATH_VAL%
if /i "%VERBOSE%"=="true" echo [WARN] Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

set "GPU_FLAG=--gpu"
if /i "%USE_GPU%"=="false" set "GPU_FLAG=--cpu"

set "LORA_PATH_FLAG="
if not "%LORA_PATH_VAL%"=="" set "LORA_PATH_FLAG=--lora-path %LORA_PATH_VAL%"

set "JFR_FLAG_LORA="
if not "%JFR_DURATION_LORA%"=="" (
  for /f "tokens=2 delims==" %%T in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "DT=%%T"
  set "JFR_TS=!DT:~0,8!-!DT:~8,6!"
  set "JFR_FLAG_LORA=-XX:StartFlightRecording=duration=%JFR_DURATION_LORA%,filename=juno-!JFR_TS!.jfr,settings=profile,dumponexit=true"
  echo [WARN] JFR enabled -- duration=%JFR_DURATION_LORA%  output=juno-!JFR_TS!.jfr
  echo [WARN] After exit: open juno-!JFR_TS!.jfr in JDK Mission Control -^> Event Browser -^> juno.LoraTrainStep
)

call :prepend_cuda_path

"%JAVA%" %JVM_BASE% -Xms512m "-Xmx%HEAP%" %JFR_FLAG_LORA% -jar "%PLAYER_JAR%" --model-path "%MODEL%" --lora --lora-rank %LORA_RANK% --lora-alpha %LORA_ALPHA% --lora-lr %LORA_LR% --lora-steps %LORA_STEPS% --max-tokens %MAX_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K% --top-p %TOP_P% %LORA_PATH_FLAG% %GPU_FLAG% %VERBOSE_FLAG%
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
echo   run.bat lora    --model-path PATH    LoRA fine-tuning REPL (adapter kept separate)
echo   run.bat test    --model-path PATH    8 smoke checks
echo.
echo   Backend flags (cluster/local/lora):
echo     --gpu          use GPU when available (default)
echo     --cpu          use CPU only
echo.
echo   Env overrides: MODEL_PATH  DTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  NODES
echo                  LORA_PATH  LORA_RANK  LORA_ALPHA  LORA_LR  LORA_STEPS  USE_GPU
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