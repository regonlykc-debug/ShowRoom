#!/bin/bash
echo "============================================"
echo "  Khalaifat Showroom AI - Starting Server"
echo "============================================"
echo ""
echo "Tablet can connect at: http://192.168.100.62:8000"
echo ""

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
