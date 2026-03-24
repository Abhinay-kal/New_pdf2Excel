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

if exist ".venv\" (
    set "VENV_DIR=.venv"
) else if exist "venv\" (
    set "VENV_DIR=venv"
) else (
    set "VENV_DIR=.venv"
    echo [Electra-Core] Creating virtual environment (%VENV_DIR%)...
    %PY_CMD% -m venv %VENV_DIR%
    if errorlevel 1 goto :error
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error

echo [Electra-Core] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [Electra-Core] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

set "PDF_PATH=%~1"
if not "%PDF_PATH%"=="" goto :run

for %%F in (*.pdf) do (
    set "PDF_PATH=%CD%\%%F"
    goto :run
)

for /f "delims=" %%I in ('powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Filter = 'PDF Files (*.pdf)|*.pdf'; $d.Title = 'Select voter-roll PDF'; if($d.ShowDialog() -eq 'OK'){ $d.FileName }"') do set "PDF_PATH=%%I"

:run
if "%PDF_PATH%"=="" (
    echo [ERROR] No PDF selected/found. Re-run and choose a PDF file.
    goto :error
)
if not exist "%PDF_PATH%" (
    echo [ERROR] PDF not found: %PDF_PATH%
    goto :error
)

echo [Electra-Core] Running extraction for: %PDF_PATH%
python main.py "%PDF_PATH%"
if errorlevel 1 goto :error

echo.
echo [Electra-Core] Completed.
pause
exit /b 0

:error
echo.
echo [ERROR] Startup failed. See messages above.
pause
exit /b 1
