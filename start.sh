#!/bin/bash
# Serves this folder over HTTP so the PDF viewer can load catalogue files.
# Opening index.html directly (file://) does NOT work — browsers block
# local PDF fetches from the file:// origin.
cd "$(dirname "$0")"
PORT=8080
echo "============================================"
echo "  Khalaifat Catalogues"
echo "============================================"
echo ""
echo "Open in a browser: http://localhost:$PORT"
echo "On the same Wi-Fi, a tablet can use: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
echo ""
python3 -m http.server "$PORT"
