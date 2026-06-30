@echo off
echo ============================================
echo   Khalaifat Showroom AI - Setup
echo ============================================

echo.
echo [1/3] Installing Python packages...
pip install fastapi uvicorn pdfplumber sentence-transformers numpy requests python-dotenv

echo.
echo [2/3] Pulling AI model (qwen2.5:7b)...
echo This may take a few minutes on first run...
ollama pull qwen2.5:7b

echo.
echo [3/3] Done! Now run: 2_index_pdfs.bat
pause
