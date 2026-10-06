@echo off
REM Patient application. Run from the repository root.
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [SPECTRA] Virtual environment not found. Create it with:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo [SPECTRA] Starting the patient application...
python -m spectra %*
if errorlevel 1 pause
endlocal
