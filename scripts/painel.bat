@echo off
REM Therapist panel. Run from the repository root.
setlocal
cd /d "%~dp0\.."
set "SPECTRA_PYTHON="

for %%E in (.venv-py312 .venv .venv312) do (
    if not defined SPECTRA_PYTHON if exist "%%E\Scripts\python.exe" (
        "%%E\Scripts\python.exe" -c "import streamlit, streamlit_webrtc" >nul 2>&1 && set "SPECTRA_PYTHON=%%E\Scripts\python.exe"
    )
)

if not defined SPECTRA_PYTHON (
    echo [SPECTRA] Virtual environment not found. Create it with:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo [SPECTRA] Starting the therapist panel...
"%SPECTRA_PYTHON%" -m streamlit run spectra\therapist_panel\app.py --server.address localhost --browser.gatherUsageStats false
if errorlevel 1 pause
endlocal
