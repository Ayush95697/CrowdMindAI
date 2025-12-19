@echo off
echo ========================================
echo   CrowdMind AI - Starting Frontend
echo ========================================
echo.

cd frontend

echo Checking if dependencies are installed...
if not exist "node_modules\" (
    echo Installing dependencies...
    echo.
    call npm install
    echo.
)

echo Starting React development server...
echo Frontend will open at: http://localhost:3000
echo.

call npm start

pause
