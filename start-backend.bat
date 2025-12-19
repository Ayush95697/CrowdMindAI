@echo off
echo ========================================
echo   CrowdMind AI - Starting Backend
echo ========================================
echo.

cd backend
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting FastAPI backend server...
echo Backend will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

pause
