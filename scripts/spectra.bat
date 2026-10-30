@echo off
REM Patient application. Run from the repository root.
setlocal
cd /d "%~dp0\.."
set "SPECTRA_PYTHON="

for %%E in (.venv-py312 .venv .venv312) do (
    if not defined SPECTRA_PYTHON if exist "%%E\Scripts\python.exe" (
        "%%E\Scripts\python.exe" -c "import mediapipe, cv2" >nul 2>&1 && set "SPECTRA_PYTHON=%%E\Scripts\python.exe"
    )
)

if not defined SPECTRA_PYTHON (
    echo [SPECTRA] No virtual environment with the dependencies was found. Create it with:
    echo     py -3.12 -m venv .venv-py312
    echo     .venv-py312\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo [SPECTRA] Starting the patient application...
"%SPECTRA_PYTHON%" -m spectra %*
if errorlevel 1 pause
endlocal
