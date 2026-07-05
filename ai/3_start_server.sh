#!/bin/bash
echo "============================================"
echo "  Khalaifat Showroom AI - Starting Server"
echo "============================================"
echo ""
echo "Serves the whole site + AI chat on one port."
echo "Tablet can connect at: http://192.168.100.58:8080"
echo ""

if [ -z "$NVIDIA_API_KEY" ]; then
    echo "ERROR: NVIDIA_API_KEY is not set. Run: export NVIDIA_API_KEY=nvapi-..."
    exit 1
fi

python3 -m uvicorn server:app --host 0.0.0.0 --port 8080
