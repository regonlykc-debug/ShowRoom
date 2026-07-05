@echo off
echo ============================================
echo   Khalaifat Showroom AI - Starting Server
echo ============================================
echo.
echo Serves the whole site + AI chat on one port.
echo Tablet can connect at: http://192.168.100.58:8080
echo.
if "%NVIDIA_API_KEY%"=="" (
    echo ERROR: NVIDIA_API_KEY is not set. Run: setx NVIDIA_API_KEY nvapi-...
    pause
    exit /b 1
)
python -m uvicorn server:app --host 0.0.0.0 --port 8080 --reload
pause
