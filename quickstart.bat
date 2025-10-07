@echo off
REM GridWorld Q-Learning Quick Start Script for Windows

echo ======================================
echo GridWorld Q-Learning Quick Start
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python 3 is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Check Python found: %PYTHON_VERSION%
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Check Virtual environment created
) else (
    echo Check Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Check Virtual environment activated
echo.

REM Install requirements
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Check Dependencies installed
echo.

REM Create output directories
echo Creating output directories...
if not exist "output\" mkdir output
if not exist "output\models\" mkdir output\models
if not exist "output\plots\" mkdir output\plots
if not exist "output\gifs\" mkdir output\gifs
if not exist "output\reports\" mkdir output\reports
type nul > output\.gitkeep
type nul > output\models\.gitkeep
type nul > output\plots\.gitkeep
type nul > output\gifs\.gitkeep
type nul > output\reports\.gitkeep
echo Check Output directories created
echo.

REM Launch Streamlit app
echo ======================================
echo Launching Streamlit app...
echo ======================================
echo.
echo The app will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py
pause