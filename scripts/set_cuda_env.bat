@echo off
setlocal enabledelayedexpansion
rem Created by Yevhen Soldatov
rem Initial implementation: 2026
rem
rem Optional: put the CUDA toolkit's DLL folders on PATH, for running a Juno jar
rem with java directly. run.bat does the same itself before every GPU run, so it
rem needs no call to this script. The folders are CUDA_PATH's, else CUDA_HOME's,
rem else the last version under the installer's default location; bin\x64 holds
rem the DLLs in the 13.x layout, bin in the 12.x one.
rem
rem Usage: call scripts\set_cuda_env.bat   (then java ... -jar juno-player-*-shaded.jar ...)

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
