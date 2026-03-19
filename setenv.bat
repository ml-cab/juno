@echo off
setlocal enabledelayedexpansion
rem Created by Yevhen Soldatov
rem Initial implementation: 2026
rem
rem Optional: set up PATH for GPU runs so jnicudart.dll can load cudart64_*.dll.
rem Call before run.bat cluster ... --gpu if CUDA is not already on PATH.
rem
rem Usage: call setenv.bat   (then run.bat cluster ... --gpu)
rem    or: setenv.bat        (sets env in this shell only)

set "CUDA_BIN="
if not "%CUDA_PATH%"=="" (
  if exist "%CUDA_PATH%\bin\x64" set "CUDA_BIN=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin"
  if "!CUDA_BIN!"=="" if exist "%CUDA_PATH%\bin" set "CUDA_BIN=%CUDA_PATH%\bin"
)
if "!CUDA_BIN!"=="" if not "%CUDA_HOME%"=="" (
  if exist "%CUDA_HOME%\bin\x64" set "CUDA_BIN=%CUDA_HOME%\bin\x64;%CUDA_HOME%\bin"
  if "!CUDA_BIN!"=="" if exist "%CUDA_HOME%\bin" set "CUDA_BIN=%CUDA_HOME%\bin"
)
if "!CUDA_BIN!"=="" (
  for /f "delims=" %%v in ('dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do set "CUDA_VER=%%v"
  if defined CUDA_VER (
    set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\!CUDA_VER!\bin"
    if exist "!CUDA_BIN!\x64" set "CUDA_BIN=!CUDA_BIN!\x64;!CUDA_BIN!"
  )
)
if not "!CUDA_BIN!"=="" (
  endlocal & set "PATH=%CUDA_BIN%;%PATH%" & echo [juno] Added CUDA bin to PATH
) else (
  endlocal & echo [juno] No CUDA found. Set CUDA_PATH or CUDA_HOME, or install CUDA to default location.
)
