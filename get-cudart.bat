@echo off
rem Created by Yevhen Soldatov
rem Initial implementation: 2026
rem
rem Fetches cudart64_12.dll (required by jnicudart.dll for GPU) without the full CUDA Toolkit.
rem Download page: https://www.dllme.com/dll/files/cudart64_12
rem
set "CACHE=%USERPROFILE%\.javacpp\cache"
set "DLL=%CACHE%\cudart64_12.dll"

if exist "%DLL%" (
  echo cudart64_12.dll already present in %CACHE%
  echo Delete it first if you want to re-download.
  exit /b 0
)

if not exist "%CACHE%" mkdir "%CACHE%"
echo Opening download page in your browser.
echo.
echo 1. Download the 64-bit cudart64_12.dll from the page that opened.
echo 2. Save it as: %DLL%
echo.
echo Or download from: https://www.dllme.com/dll/files/cudart64_12
start "" "https://www.dllme.com/dll/files/cudart64_12"
exit /b 0
