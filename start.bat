@echo off
REM Serves this folder over HTTP so the PDF viewer can load catalogue files.
REM Opening index.html directly (file://) does NOT work — browsers block
REM local PDF fetches from the file:// origin.
REM
REM NOTE: ai\3_start_server.bat now serves the whole site (this same content)
REM PLUS the AI chat, on the same port 8080 — that's the one script to run for
REM normal/customer use. Only use this script for a quick static-only preview
REM with the AI chat disabled, and don't run both at once (port 8080 conflict).
cd /d "%~dp0"
set PORT=8080
echo ============================================
echo   Khalaifat Catalogues (static preview, no AI)
echo ============================================
echo.
echo Open in a browser: http://localhost:%PORT%
echo.
python -m http.server %PORT%
pause
