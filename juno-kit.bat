@echo off
rem juno-kit.bat - hardware validation kit launcher (juno-kit.sh elsewhere)
rem
rem Builds Juno and the kit with Maven, detects what this machine can run, runs
rem Junos checks and readings, and packs the results into one zip to send back.
rem Needs JDK 25 or later and Maven 3.9 or later; nothing else.
rem
rem   juno-kit.bat                 asks what to run
rem   juno-kit.bat --suite quick   TinyLlama only
rem   juno-kit.bat --help          every option
setlocal
set "DIR=%~dp0"
set "DIR=%DIR:~0,-1%"
if defined JAVA_HOME (set "JAVA=%JAVA_HOME%\bin\java.exe") else (set "JAVA=java")
"%JAVA%" "-Djuno.kit.root=%DIR%" "%DIR%\juno-kit\src\main\java\cab\ml\juno\kit\KitBootstrap.java" %*
exit /b %ERRORLEVEL%
