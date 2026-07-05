#!/bin/bash
# Serves this folder over HTTP so the PDF viewer can load catalogue files.
# Opening index.html directly (file://) does NOT work — browsers block
# local PDF fetches from the file:// origin.
#
# NOTE: ai/3_start_server.sh now serves the whole site (this same content)
# PLUS the AI chat, on the same port 8080 — that's the one script to run for
# normal/customer use. Only use this script for a quick static-only preview
# with the AI chat disabled, and don't run both at once (port 8080 conflict).
cd "$(dirname "$0")"
PORT=8080
echo "============================================"
echo "  Khalaifat Catalogues (static preview, no AI)"
echo "============================================"
echo ""
echo "Open in a browser: http://localhost:$PORT"
echo "On the same Wi-Fi, a tablet can use: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
echo ""
python3 -m http.server "$PORT"
