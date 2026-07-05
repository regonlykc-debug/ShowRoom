#!/bin/bash
set -e
echo "============================================"
echo "  Khalaifat Showroom AI - Setup"
echo "============================================"

echo ""
echo "[1/2] Installing Python packages..."
# --break-system-packages: needed on Debian/Ubuntu-managed Python (PEP 668)
# where a bare `pip install` refuses to run at all. This machine is a
# dedicated kiosk box, not a shared system, so this is safe here.
if ! pip install fastapi uvicorn sentence-transformers numpy openai pdfplumber 2>/tmp/pip_error.log; then
    if grep -q "externally-managed-environment" /tmp/pip_error.log; then
        pip install --break-system-packages fastapi uvicorn sentence-transformers numpy openai pdfplumber
    else
        cat /tmp/pip_error.log >&2
        exit 1
    fi
fi
rm -f /tmp/pip_error.log

echo ""
echo "[2/2] Done! Before starting the server, get a free API key at"
echo "build.nvidia.com and set it:"
echo "  export NVIDIA_API_KEY=nvapi-..."
echo "Then run: bash 2_index_products.sh (and bash 2b_index_pdf_brands.sh once)"
