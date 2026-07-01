@echo off
echo ============================================
echo   Khalaifat Showroom AI - Starting Server
echo ============================================
echo.
echo Make sure Ollama is running first!
echo Tablet can connect at: http://192.168.100.62:8000
echo.
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
