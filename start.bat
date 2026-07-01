@echo off
REM Serves this folder over HTTP so the PDF viewer can load catalogue files.
REM Opening index.html directly (file://) does NOT work — browsers block
REM local PDF fetches from the file:// origin.
cd /d "%~dp0"
set PORT=8080
echo ============================================
echo   Khalaifat Catalogues
echo ============================================
echo.
echo Open in a browser: http://localhost:%PORT%
echo.
python -m http.server %PORT%
pause
