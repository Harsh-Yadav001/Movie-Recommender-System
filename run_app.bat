@echo off
set "APP_DIR=%~dp0"

if not exist "%APP_DIR%venv\Scripts\python.exe" (
    echo Project virtual environment not found at "%APP_DIR%venv".
    echo Please create the virtual environment and install the project dependencies.
    pause
    exit /b 1
)

"%APP_DIR%venv\Scripts\python.exe" -m streamlit run "%APP_DIR%recomender_system\app.py"
if errorlevel 1 pause
