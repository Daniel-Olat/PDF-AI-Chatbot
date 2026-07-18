@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%.venv\Scripts\python.exe" -u "%SCRIPT_DIR%rag-project\rag.py"
if errorlevel 1 exit /b %errorlevel%
