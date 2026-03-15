@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem  hyper-stack-4j — Windows runtime launcher (no Maven required)
rem  Uses pre-built shade jars from target/.
rem  Build first with:
rem      mvn clean package -DskipTests
rem      or: hyper.sh build
rem
rem  Requires: JDK 21+
rem  Runs on:  Windows (cmd.exe, PowerShell -> cmd)
rem ============================================================================

rem --- Script directory -------------------------------------------------------
set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"

set "PLAYER_JAR=%DIR%\player\target\player.jar"
set "LIVE_JAR=%DIR%\integration\target\integration.jar"

rem --- Java discovery ---------------------------------------------------------
call :find_java
if errorlevel 1 goto :eof

rem --- Java version check -----------------------------------------------------
call :check_java_version
if errorlevel 1 goto :eof

rem --- Dispatch ---------------------------------------------------------------
if "%~1"=="" goto :usage
set "CMD=%~1"
shift

if /i "%CMD%"=="cluster" goto :cluster
if /i "%CMD%"=="console" goto :console
if /i "%CMD%"=="live" goto :live
goto :usage

rem ============================================================================
rem  Commands
rem ============================================================================

:cluster
rem 3-node distributed cluster + interactive REPL (forked JVM nodes)
set "MODEL=%MODEL_PATH%"
set "DTYPE=%DTYPE%"
set "MAX_TOKENS=%MAX_TOKENS%"
set "TEMPERATURE=%TEMPERATURE%"
set "TOP_K=%TOP_K%"
set "TOP_P=%TOP_P%"
set "HEAP=%HEAP%"
set "VERBOSE=false"

if "%DTYPE%"=="" set "DTYPE=FLOAT16"
if "%MAX_TOKENS%"=="" set "MAX_TOKENS=200"
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.6"
if "%TOP_K%"=="" set "TOP_K=20"
if "%TOP_P%"=="" set "TOP_P=0.95"
if "%HEAP%"=="" set "HEAP=4g"

:cluster_parse
if "%~1"=="" goto :cluster_done

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--dtype" (
  set "DTYPE=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--max-tokens" (
  set "MAX_TOKENS=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--top-k" (
  set "TOP_K=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--top-p" (
  set "TOP_P=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--temperature" (
  set "TEMPERATURE=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift & shift
  goto :cluster_parse
)
if /i "%~1"=="--float16" (
  set "DTYPE=FLOAT16"
  shift
  goto :cluster_parse
)
if /i "%~1"=="--fp16" (
  set "DTYPE=FLOAT16"
  shift
  goto :cluster_parse
)
if /i "%~1"=="--float32" (
  set "DTYPE=FLOAT32"
  shift
  goto :cluster_parse
)
if /i "%~1"=="--int8" (
  set "DTYPE=INT8"
  shift
  goto :cluster_parse
)
if /i "%~1"=="--verbose" (
  set "VERBOSE=true"
  shift
  goto :cluster_parse
)
if /i "%~1"=="-v" (
  set "VERBOSE=true"
  shift
  goto :cluster_parse
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat cluster --model-path ^<path-to-model.gguf^> [flags]
  echo      or: set MODEL_PATH=^<path^> ^&^& run.bat cluster [flags]
  echo      or: run.bat cluster ^<path-to-model.gguf^> [flags]
  echo.
  echo   Starts a 3-node cluster ^(one forked JVM per node^) and an interactive REPL.
  echo   Each node serves gRPC on localhost:19092-19094.
  echo.
  echo   Required:
  echo     --model-path PATH           Path to a GGUF model file
  echo                                  or set MODEL_PATH env var
  echo.
  echo   Activation dtype:
  echo     --dtype FLOAT32^|FLOAT16^|INT8   wire format between nodes  ^(default FLOAT16^)
  echo     --float16 / --fp16              shorthand for FLOAT16
  echo     --float32                       lossless, debug/reference
  echo     --int8                          ~4x smaller, ~1%% relative error
  echo.
  echo   Generation:
  echo     --max-tokens N              max tokens per response   ^(default 200^)
  echo     --temperature F             sampling temperature      ^(default 0.6^)
  echo     --top-k N                   top-K sampling cutoff     ^(default 20, 0=disabled^)
  echo     --top-p F                   top-p nucleus sampling    ^(default 0.95, 0=disabled^)
  echo.
  echo   JVM:
  echo     --heap SIZE                 JVM heap e.g. 4g 8g 16g   ^(default 4g^)
  echo.
  echo   Logging:
  echo     --verbose / -v              show gRPC and node logs
  echo.
  goto :eof
)

rem Positional model path support: first non-flag argument
if "%MODEL%"=="" if exist "%~1" (
  set "MODEL=%~1"
  shift
  goto :cluster_parse
)

echo Unknown cluster flag: %1
echo Run: run.bat cluster --help
exit /b 1

:cluster_done
if "%MODEL%"=="" (
  echo Model path is required.
  echo   Usage: run.bat cluster --model-path ^<path-to-model.gguf^>
  echo      or: set MODEL_PATH=^<path^> ^&^& run.bat cluster
  exit /b 1
)
if not exist "%MODEL%" (
  echo Model file not found: "%MODEL%"
  exit /b 1
)
if not exist "%PLAYER_JAR%" (
  echo Player jar not found: "%PLAYER_JAR%"
  echo   Build first: mvn clean package -DskipTests
  exit /b 1
)

echo Starting 3-node cluster  ^(dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  heap=%HEAP%^)
if /i "%VERBOSE%"=="true" echo Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

"%JAVA_CMD%" ^
  --enable-preview ^
  --enable-native-access=ALL-UNNAMED ^
  --add-opens java.base/java.lang=ALL-UNNAMED ^
  --add-opens java.base/java.nio=ALL-UNNAMED ^
  -XX:+UseG1GC ^
  -XX:+AlwaysPreTouch ^
  -Xms512m -Xmx%HEAP% ^
  -jar "%PLAYER_JAR%" ^
  --model-path "%MODEL%" ^
  --dtype "%DTYPE%" ^
  --max-tokens %MAX_TOKENS% ^
  --temperature %TEMPERATURE% ^
  --top-k %TOP_K% ^
  --top-p %TOP_P% ^
  %VERBOSE_FLAG%
goto :eof

:console
rem Single-JVM in-process REPL ^(no forked nodes, fastest startup^)
set "MODEL=%MODEL_PATH%"
set "DTYPE=%DTYPE%"
set "MAX_TOKENS=%MAX_TOKENS%"
set "TEMPERATURE=%TEMPERATURE%"
set "HEAP=%HEAP%"
set "NODES=%NODES%"
set "VERBOSE=false"

if "%DTYPE%"=="" set "DTYPE=FLOAT16"
if "%MAX_TOKENS%"=="" set "MAX_TOKENS=200"
if "%TEMPERATURE%"=="" set "TEMPERATURE=0.6"
if "%HEAP%"=="" set "HEAP=4g"
if "%NODES%"=="" set "NODES=3"
if "%TOP_K%"=="" set "TOP_K=20"
if "%TOP_P%"=="" set "TOP_P=0.95"

:console_parse
if "%~1"=="" goto :console_done

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--dtype" (
  set "DTYPE=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--max-tokens" (
  set "MAX_TOKENS=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--temperature" (
  set "TEMPERATURE=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--top-k" (
  set "TOP_K=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--top-p" (
  set "TOP_P=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--nodes" (
  set "NODES=%~2"
  shift & shift
  goto :console_parse
)
if /i "%~1"=="--float16" (
  set "DTYPE=FLOAT16"
  shift
  goto :console_parse
)
if /i "%~1"=="--fp16" (
  set "DTYPE=FLOAT16"
  shift
  goto :console_parse
)
if /i "%~1"=="--float32" (
  set "DTYPE=FLOAT32"
  shift
  goto :console_parse
)
if /i "%~1"=="--int8" (
  set "DTYPE=INT8"
  shift
  goto :console_parse
)
if /i "%~1"=="--verbose" (
  set "VERBOSE=true"
  shift
  goto :console_parse
)
if /i "%~1"=="-v" (
  set "VERBOSE=true"
  shift
  goto :console_parse
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat console --model-path ^<path-to-model.gguf^> [flags]
  echo      or: set MODEL_PATH=^<path^> ^&^& run.bat console [flags]
  echo      or: run.bat console ^<path-to-model.gguf^> [flags]
  echo.
  echo   Runs all transformer nodes in-process in a single JVM ^- no forking,
  echo   no gRPC sockets. Fastest startup. Use this for everyday experimentation.
  echo.
  echo   Required:
  echo     --model-path PATH           Path to a GGUF model file
  echo                                  or set MODEL_PATH env var
  echo.
  echo   Activation dtype:
  echo     --dtype FLOAT32^|FLOAT16^|INT8   ^(default FLOAT16^)
  echo     --float16 / --fp16
  echo     --float32
  echo     --int8
  echo.
  echo   Generation:
  echo     --max-tokens N              ^(default 200^)
  echo     --temperature F             ^(default 0.6^)
  echo     --top-k N                   top-K sampling cutoff     ^(default 20, 0=disabled^)
  echo     --top-p F                   top-p nucleus sampling    ^(default 0.95, 0=disabled^)
  echo.
  echo   Pipeline:
  echo     --nodes N                   number of in-process shards  ^(default 3^)
  echo.
  echo   JVM:
  echo     --heap SIZE                 e.g. 4g 8g 16g               ^(default 4g^)
  echo.
  echo   Logging:
  echo     --verbose / -v
  echo.
  goto :eof
)

rem Positional model path support: first non-flag argument
if "%MODEL%"=="" if exist "%~1" (
  set "MODEL=%~1"
  shift
  goto :console_parse
)

echo Unknown console flag: %1
echo Run: run.bat console --help
exit /b 1

:console_done
if "%MODEL%"=="" (
  echo Model path is required.
  echo   Usage: run.bat console --model-path ^<path-to-model.gguf^>
  echo      or: set MODEL_PATH=^<path^> ^&^& run.bat console
  exit /b 1
)
if not exist "%MODEL%" (
  echo Model file not found: "%MODEL%"
  exit /b 1
)
if not exist "%PLAYER_JAR%" (
  echo Player jar not found: "%PLAYER_JAR%"
  echo   Build first: mvn clean package -DskipTests
  exit /b 1
)

echo Starting local in-process console  ^(dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  nodes=%NODES%  heap=%HEAP%^)
if /i "%VERBOSE%"=="true" echo Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

"%JAVA_CMD%" ^
  --enable-preview ^
  --enable-native-access=ALL-UNNAMED ^
  --add-opens java.base/java.lang=ALL-UNNAMED ^
  --add-opens java.base/java.nio=ALL-UNNAMED ^
  -XX:+UseG1GC ^
  -XX:+AlwaysPreTouch ^
  -Xms512m -Xmx%HEAP% ^
  -jar "%PLAYER_JAR%" ^
  --model-path "%MODEL%" ^
  --dtype "%DTYPE%" ^
  --max-tokens %MAX_TOKENS% ^
  --temperature %TEMPERATURE% ^
  --top-k %TOP_K% ^
  --top-p %TOP_P% ^
  --nodes %NODES% ^
  --local ^
  %VERBOSE_FLAG%
goto :eof

:live
rem ModelLiveRunner: 6 real-model smoke checks, exits 0/1
set "MODEL=%MODEL_PATH%"
set "HEAP=%HEAP%"
if "%HEAP%"=="" set "HEAP=4g"

:live_parse
if "%~1"=="" goto :live_done

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift & shift
  goto :live_parse
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift & shift
  goto :live_parse
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat live --model-path ^<path-to-model.gguf^> [flags]
  echo      or: set MODEL_PATH=^<path^> ^&^& run.bat live [flags]
  echo      or: run.bat live ^<path-to-model.gguf^>
  echo.
  echo   Runs ModelLiveRunner ^- 6 automated real-model checks.
  echo   Exits 0 if all pass, 1 if any fail.
  echo.
  echo   Required:
  echo     --model-path PATH  or  MODEL_PATH env var  or positional arg
  echo.
  echo   JVM:
  echo     --heap SIZE        e.g. 4g 8g 16g  ^(default 4g^)
  echo.
  goto :eof
)

if "%MODEL%"=="" if exist "%~1" (
  set "MODEL=%~1"
  shift
  goto :live_parse
)

echo Unknown live flag: %1
echo Run: run.bat live --help
exit /b 1

:live_done
if "%MODEL%"=="" (
  echo Model path is required.
  echo   Usage: set MODEL_PATH=^<path^> ^&^& run.bat live
  echo      or: run.bat live ^<path-to-model.gguf^>
  echo      or: run.bat live --model-path ^<path-to-model.gguf^>
  exit /b 1
)
if not exist "%MODEL%" (
  echo Model file not found: "%MODEL%"
  exit /b 1
)
if not exist "%LIVE_JAR%" (
  echo Integration jar not found: "%LIVE_JAR%"
  echo   Build first: mvn clean package -DskipTests
  exit /b 1
)

echo Running ModelLiveRunner  ^(model=%MODEL%  heap=%HEAP%^)
echo.

"%JAVA_CMD%" ^
  --enable-preview ^
  --enable-native-access=ALL-UNNAMED ^
  --add-opens java.base/java.lang=ALL-UNNAMED ^
  --add-opens java.base/java.nio=ALL-UNNAMED ^
  -XX:+UseG1GC ^
  -XX:+AlwaysPreTouch ^
  -Xms512m -Xmx%HEAP% ^
  -jar "%LIVE_JAR%" ^
  "%MODEL%"
goto :eof

rem ============================================================================
rem  Usage
rem ============================================================================

:usage
echo.
echo hyper-stack-4j runtime launcher ^(Windows, no Maven — uses pre-built jars^)
echo   Java:      %JAVA_CMD%
echo   Player jar:%PLAYER_JAR%
echo   Live jar:  %LIVE_JAR%
echo.
echo   Build jars first ^(one time^):
echo     mvn clean package -DskipTests
echo     hyper.sh build
echo.
echo   run.bat cluster --model-path PATH    3-node cluster ^+ REPL  ^(forked JVM nodes^)
echo   run.bat cluster --help               all cluster flags
echo.
echo   run.bat console --model-path PATH    in-process REPL  ^(single JVM, fast startup^)
echo   run.bat console --help               all console flags
echo.
echo   run.bat live --model-path PATH       6 real-model smoke checks, exits 0/1
echo   run.bat live ^<path-to-model.gguf^>        model as positional arg
echo   run.bat live --help                  all live flags
echo.
echo   Flags common to cluster and console:
echo     --dtype FLOAT32^|FLOAT16^|INT8    activation wire format   ^(default FLOAT16^)
echo     --float16 / --fp16                shorthand
echo     --float32                         lossless reference / debug
echo     --int8                            maximum compression
echo     --max-tokens N                    max tokens per response  ^(default 200^)
echo     --temperature F                   sampling temperature      ^(default 0.6^)
echo     --top-k N                         top-K sampling cutoff     ^(default 20, 0=disabled^)
echo     --top-p F                         top-p nucleus sampling    ^(default 0.95, 0=disabled^)
echo     --heap SIZE                       JVM heap e.g. 4g 8g      ^(default 4g^)
echo     --verbose / -v                    show gRPC / node logs
echo.
echo   console only:
echo     --nodes N                         in-process shard count   ^(default 3^)
echo.
echo   Environment overrides:
echo     MODEL_PATH  DTYPE  MAX_TOKENS  TEMPERATURE  TOP_K  TOP_P  HEAP  NODES
echo.
echo   Examples:
echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat cluster
echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat cluster --float32 --heap 8g --verbose
echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat console --temperature 0.3 --max-tokens 512
echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat console --nodes 1
echo     set MODEL_PATH=C:\models\tiny.gguf ^&^& run.bat live
echo     run.bat live C:\models\tiny.gguf --heap 8g
echo.
goto :eof

rem ============================================================================
rem  Helpers
rem ============================================================================

:find_java
if not "%JAVA_HOME%"=="" (
  if exist "%JAVA_HOME%\bin\java.exe" (
    set "JAVA_CMD=%JAVA_HOME%\bin\java.exe"
    exit /b 0
  )
)

where java >nul 2>&1
if %errorlevel%==0 (
  set "JAVA_CMD=java"
  exit /b 0
)

echo JDK 21+ not found. Install a JDK and set JAVA_HOME.
exit /b 1

:check_java_version
set "JAVA_VER="
for /f "tokens=3 delims= " %%v in ('"%JAVA_CMD%" -version 2^>^&1 ^| findstr /i "version"') do (
  set "JAVA_VER=%%v"
  goto :have_ver
)

:have_ver
if "%JAVA_VER%"=="" (
  echo Warning: unable to determine Java version. Continuing.
  exit /b 0
)

set "JAVA_VER=%JAVA_VER:"=%"
set "JAVA_MAJOR="
for /f "tokens=1 delims=." %%m in ("%JAVA_VER%") do set "JAVA_MAJOR=%%m"

if "%JAVA_MAJOR%"=="" (
  echo Warning: unable to parse Java version "%JAVA_VER%". Continuing.
  exit /b 0
)

set /a JAVA_MAJOR_NUM=%JAVA_MAJOR% 2>nul
if %JAVA_MAJOR_NUM% LSS 21 (
  echo JDK 21+ required ^(found: %JAVA_VER%^).
  exit /b 1
)

exit /b 0

@echo off
rem ============================================================================
rem hyper-stack-4j — Windows launcher (cmd.exe)
rem
rem 1) player cluster     : run.bat cluster  --model-path C:\path\model.gguf
rem 2) player local REPL  : run.bat console  --model-path C:\path\model.gguf
rem 3) integration live   : run.bat live     C:\path\model.gguf
rem
rem Uses pre-built shade jars from target/. Build first with:
rem   mvn clean package -DskipTests   or   ./hyper.sh build
rem Requires: JDK 21+
rem ============================================================================

setlocal enabledelayedexpansion

set "DIR=%~dp0"
set "PLAYER_JAR=%DIR%player\target\player.jar"
set "LIVE_JAR=%DIR%integration\target\integration.jar"

rem --- Colour-ish prefixes (plain text, no ANSI for maximum compatibility) ----
set "P_INFO=[INFO]"
set "P_WARN=[WARN]"
set "P_ERR=[ERR]"

rem --- Java discovery ---------------------------------------------------------

set "JAVA="

if defined JAVA_HOME (
  if exist "%JAVA_HOME%\bin\java.exe" (
    set "JAVA=%JAVA_HOME%\bin\java.exe"
  )
)

if not defined JAVA (
  for /f "usebackq tokens=* delims=" %%J in (`where java 2^>nul`) do (
    if not defined JAVA (
      set "JAVA=%%J"
    )
  )
)

if not defined JAVA (
  echo %P_ERR% JDK 21+ not found. Install from https://adoptium.net and set JAVA_HOME.
  exit /b 1
)

rem --- Java version check (require 21+) --------------------------------------

set "JAVAVER_RAW="
for /f "usebackq tokens=3 delims= " %%V in (`"%JAVA%" -version 2^>^&1 ^| findstr /i "version"`) do (
  set "JAVAVER_RAW=%%V"
  goto :have_ver
)
:have_ver
if not defined JAVAVER_RAW (
  echo %P_WARN% Could not parse java version; assuming it is >= 21.
) else (
  set "JAVAVER=%JAVAVER_RAW:"=%"
  for /f "tokens=1 delims=." %%M in ("%JAVAVER%") do set "JAVAMAJOR=%%M"
  if not "%JAVAMAJOR%"=="" (
    rem numeric compare
    set /a _tmp=%JAVAMAJOR% 1>nul 2>nul
    if %JAVAMAJOR% LSS 21 (
      echo %P_ERR% JDK 21+ required (found: %JAVAMAJOR%).  JAVA_HOME=%JAVA_HOME%
      exit /b 1
    )
  )
)

rem --- Jar existence check ----------------------------------------------------

if not exist "%PLAYER_JAR%" (
  echo %P_ERR% player jar not found: "%PLAYER_JAR%"
  echo        Build first: mvn clean package -DskipTests
  echo                 or: ./hyper.sh build
  exit /b 1
)

if not exist "%LIVE_JAR%" (
  echo %P_ERR% integration jar not found: "%LIVE_JAR%"
  echo        Build first: mvn clean package -DskipTests
  echo                 or: ./hyper.sh build
  exit /b 1
)

rem --- Common JVM flags -------------------------------------------------------

set "JVM_BASE=--enable-preview --enable-native-access=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED -XX:+UseG1GC -XX:+AlwaysPreTouch"

rem --- Usage ------------------------------------------------------------------

:usage
if "%~1"=="" (
  echo.
  echo hyper-stack-4j runtime launcher ^(Windows batch, uses pre-built jars^)
  echo   Java:        %JAVA%
  echo   player jar:  %PLAYER_JAR%
  echo   live jar:    %LIVE_JAR%
  echo.
  echo   Build jars first ^(one time^):
  echo     mvn clean package -DskipTests
  echo     ./hyper.sh build
  echo.
  echo   1^)^ player cluster  ^(forked JVM nodes, REPL^):
  echo     run.bat cluster --model-path C:\path\to\model.gguf  [flags]
  echo.
  echo   2^)^ player local REPL ^(single JVM, fast startup^):
  echo     run.bat console --model-path C:\path\to\model.gguf  [flags]
  echo.
  echo   3^)^ integration live checks:
  echo     run.bat live C:\path\to\model.gguf
  echo     run.bat live --model-path C:\path\to\model.gguf
  echo.
  echo   Shared flags for cluster/console:
  echo     --dtype FLOAT32^|FLOAT16^|INT8   ^(default FLOAT16^)
  echo     --float16  ^| --fp16
  echo     --float32
  echo     --int8
  echo     --max-tokens N                  ^(default 200^)
  echo     --temperature F                 ^(default 0.7^)
  echo     --heap SIZE                     ^(default 4g^)
  echo     --verbose  ^| -v
  echo.
  echo   console only:
  echo     --nodes N                       ^(default 3^)
  echo.
  echo   Env vars:
  echo     MODEL_PATH  DTYPE  MAX_TOKENS  TEMPERATURE  HEAP  NODES
  echo.
  goto :eof
)

set "CMD=%~1"
shift

if /i "%CMD%"=="cluster" goto :cluster
if /i "%CMD%"=="console" goto :console
if /i "%CMD%"=="live"    goto :live

goto :usage

rem --- cluster: 3-node cluster + REPL -----------------------------------------

:cluster
set "MODEL=%MODEL_PATH%"
set "DTYPE=%DTYPE%"
if not defined DTYPE set "DTYPE=FLOAT16"
set "MAX_TOKENS=%MAX_TOKENS%"
if not defined MAX_TOKENS set "MAX_TOKENS=200"
set "TEMPERATURE=%TEMPERATURE%"
if not defined TEMPERATURE set "TEMPERATURE=0.7"
set "HEAP=%HEAP%"
if not defined HEAP set "HEAP=4g"
set "VERBOSE=false"

:cluster_args
if "%~1"=="" goto :cluster_run

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift
  shift
  goto :cluster_args
)
if /i "%~1"=="--dtype" (
  set "DTYPE=%~2"
  shift
  shift
  goto :cluster_args
)
if /i "%~1"=="--max-tokens" (
  set "MAX_TOKENS=%~2"
  shift
  shift
  goto :cluster_args
)
if /i "%~1"=="--temperature" (
  set "TEMPERATURE=%~2"
  shift
  shift
  goto :cluster_args
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift
  shift
  goto :cluster_args
)
if /i "%~1"=="--float16" (
  set "DTYPE=FLOAT16"
  shift
  goto :cluster_args
)
if /i "%~1"=="--fp16" (
  set "DTYPE=FLOAT16"
  shift
  goto :cluster_args
)
if /i "%~1"=="--float32" (
  set "DTYPE=FLOAT32"
  shift
  goto :cluster_args
)
if /i "%~1"=="--int8" (
  set "DTYPE=INT8"
  shift
  goto :cluster_args
)
if /i "%~1"=="--verbose" (
  set "VERBOSE=true"
  shift
  goto :cluster_args
)
if /i "%~1"=="-v" (
  set "VERBOSE=true"
  shift
  goto :cluster_args
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat cluster --model-path C:\path\to\model.gguf [flags]
  echo          MODEL_PATH=C:\path\to\model.gguf run.bat cluster [flags]
  echo.
  echo   Starts a 3-node cluster and interactive REPL.
  echo.
  goto :eof
)

echo %P_ERR% Unknown cluster flag: %~1
echo        Run: run.bat cluster --help
exit /b 1

:cluster_run
if not defined MODEL (
  echo %P_ERR% Model path is required.
  echo        Usage: run.bat cluster --model-path C:\path\to\model.gguf
  echo           or: set MODEL_PATH=... ^&^& run.bat cluster
  exit /b 1
)
if not exist "%MODEL%" (
  echo %P_ERR% Model file not found: "%MODEL%"
  exit /b 1
)

echo %P_WARN% Starting 3-node cluster  ^(dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  heap=%HEAP%^)
if /i "%VERBOSE%"=="true" echo %P_WARN% Verbose mode ON
echo %P_WARN% Ctrl-C to stop all nodes and exit
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

"%JAVA%" %JVM_BASE% -Xms512m -Xmx%HEAP% -jar "%PLAYER_JAR%" --model-path "%MODEL%" --dtype "%DTYPE%" --max-tokens "%MAX_TOKENS%" --temperature "%TEMPERATURE%" %VERBOSE_FLAG%
goto :eof

rem --- console: in-process REPL (single JVM) ----------------------------------

:console
set "MODEL=%MODEL_PATH%"
set "DTYPE=%DTYPE%"
if not defined DTYPE set "DTYPE=FLOAT16"
set "MAX_TOKENS=%MAX_TOKENS%"
if not defined MAX_TOKENS set "MAX_TOKENS=200"
set "TEMPERATURE=%TEMPERATURE%"
if not defined TEMPERATURE set "TEMPERATURE=0.7"
set "HEAP=%HEAP%"
if not defined HEAP set "HEAP=4g"
set "NODES=%NODES%"
if not defined NODES set "NODES=3"
set "VERBOSE=false"

:console_args
if "%~1"=="" goto :console_run

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--dtype" (
  set "DTYPE=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--max-tokens" (
  set "MAX_TOKENS=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--temperature" (
  set "TEMPERATURE=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--nodes" (
  set "NODES=%~2"
  shift
  shift
  goto :console_args
)
if /i "%~1"=="--float16" (
  set "DTYPE=FLOAT16"
  shift
  goto :console_args
)
if /i "%~1"=="--fp16" (
  set "DTYPE=FLOAT16"
  shift
  goto :console_args
)
if /i "%~1"=="--float32" (
  set "DTYPE=FLOAT32"
  shift
  goto :console_args
)
if /i "%~1"=="--int8" (
  set "DTYPE=INT8"
  shift
  goto :console_args
)
if /i "%~1"=="--verbose" (
  set "VERBOSE=true"
  shift
  goto :console_args
)
if /i "%~1"=="-v" (
  set "VERBOSE=true"
  shift
  goto :console_args
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat console --model-path C:\path\to\model.gguf [flags]
  echo          MODEL_PATH=C:\path\to\model.gguf run.bat console [flags]
  echo.
  echo   Runs all transformer nodes in-process in a single JVM.
  echo.
  goto :eof
)

echo %P_ERR% Unknown console flag: %~1
echo        Run: run.bat console --help
exit /b 1

:console_run
if not defined MODEL (
  echo %P_ERR% Model path is required.
  echo        Usage: run.bat console --model-path C:\path\to\model.gguf
  echo           or: set MODEL_PATH=... ^&^& run.bat console
  exit /b 1
)
if not exist "%MODEL%" (
  echo %P_ERR% Model file not found: "%MODEL%"
  exit /b 1
)

echo %P_INFO% Starting local in-process console  ^(dtype=%DTYPE%  max_tokens=%MAX_TOKENS%  temperature=%TEMPERATURE%  nodes=%NODES%  heap=%HEAP%^)
if /i "%VERBOSE%"=="true" echo %P_WARN% Verbose mode ON
echo.

set "VERBOSE_FLAG="
if /i "%VERBOSE%"=="true" set "VERBOSE_FLAG=--verbose"

"%JAVA%" %JVM_BASE% -Xms512m -Xmx%HEAP% -jar "%PLAYER_JAR%" --model-path "%MODEL%" --dtype "%DTYPE%" --max-tokens "%MAX_TOKENS%" --temperature "%TEMPERATURE%" --nodes "%NODES%" --local %VERBOSE_FLAG%
goto :eof

rem --- live: ModelLiveRunner (integration) ------------------------------------

:live
set "MODEL=%MODEL_PATH%"
set "HEAP=%HEAP%"
if not defined HEAP set "HEAP=4g"

:live_args
if "%~1"=="" goto :live_run

if /i "%~1"=="--model-path" (
  set "MODEL=%~2"
  shift
  shift
  goto :live_args
)
if /i "%~1"=="--heap" (
  set "HEAP=%~2"
  shift
  shift
  goto :live_args
)
if /i "%~1"=="--help" (
  echo.
  echo   Usage: run.bat live --model-path C:\path\to\model.gguf [flags]
  echo          MODEL_PATH=C:\path\to\model.gguf run.bat live [flags]
  echo          run.bat live C:\path\to\model.gguf
  echo.
  echo   Runs ModelLiveRunner — 6 automated real-model checks.
  echo.
  goto :eof
)

rem Positional model path if file exists
if not defined MODEL if exist "%~1" (
  set "MODEL=%~1"
  shift
  goto :live_args
)

echo %P_ERR% Unknown live flag: %~1
echo        Run: run.bat live --help
exit /b 1

:live_run
if not defined MODEL (
  echo %P_ERR% Model path is required.
  echo        Usage: MODEL_PATH=C:\path\to\model.gguf run.bat live
  echo           or: run.bat live C:\path\to\model.gguf
  echo           or: run.bat live --model-path C:\path\to\model.gguf
  exit /b 1
)
if not exist "%MODEL%" (
  echo %P_ERR% Model file not found: "%MODEL%"
  exit /b 1
)

echo %P_INFO% Running ModelLiveRunner  ^(model=%~nxMODEL%  heap=%HEAP%^)
echo.

"%JAVA%" %JVM_BASE% -Xms512m -Xmx%HEAP% -jar "%LIVE_JAR%" "%MODEL%"
goto :eof

