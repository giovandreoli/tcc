@echo off
REM Therapist panel. Run from the repository root.
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

echo [SPECTRA] Starting the therapist panel...
python -m streamlit run spectra\therapist_panel\app.py
if errorlevel 1 pause
endlocal
