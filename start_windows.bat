@echo off
setlocal enableextensions

cd /d "%~dp0"
echo [Electra-Core] Starting setup...

where python >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=py -3"
    ) else (
        echo [ERROR] Python is not installed or not in PATH.
        echo Install Python 3 and re-run this script.
        pause
        exit /b 1
    )
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in %CD%
    pause
    exit /b 1
)

if not exist "venv\" (
    echo [Electra-Core] Creating virtual environment (venv)...
    %PY_CMD% -m venv venv
    if errorlevel 1 goto :error
)

call "venv\Scripts\activate.bat"
if errorlevel 1 goto :error

echo [Electra-Core] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [Electra-Core] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [Electra-Core] Launching dashboard...
python -m streamlit run app.py
if errorlevel 1 goto :error

exit /b 0

:error
echo.
echo [ERROR] Startup failed. See messages above.
pause
exit /b 1
