@echo off
echo ========================================
echo   CrowdMind AI - Full Stack Startup
echo ========================================
echo.
echo Starting backend and frontend servers...
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.

start "CrowdMind AI - Backend" cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 5 /nobreak

start "CrowdMind AI - Frontend" cmd /k "cd frontend && npm start"

echo.
echo Both servers are starting in separate windows...
echo Close this window when done.
echo.
pause
