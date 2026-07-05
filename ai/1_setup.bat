@echo off
echo ============================================
echo   Khalaifat Showroom AI - Setup
echo ============================================

echo.
echo [1/2] Installing Python packages...
pip install fastapi uvicorn sentence-transformers numpy openai pdfplumber python-dotenv

echo.
echo [2/2] Done! Before starting the server, get a free API key at
echo build.nvidia.com and set it:
echo   setx NVIDIA_API_KEY nvapi-...
echo Then run: 2_index_products.bat (and 2b_index_pdf_brands.bat once)
pause
