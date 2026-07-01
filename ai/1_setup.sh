#!/bin/bash
echo "============================================"
echo "  Khalaifat Showroom AI - Setup"
echo "============================================"

echo ""
echo "[1/4] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo ""
echo "[2/4] Installing Python packages..."
pip install fastapi uvicorn pdfplumber sentence-transformers numpy requests

echo ""
echo "[3/4] Pulling AI model (qwen2.5:7b)..."
echo "This may take a few minutes on first run..."
ollama pull qwen2.5:7b

echo ""
echo "[4/4] Done! Now run: bash 2_index_pdfs.sh"
