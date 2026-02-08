#!/bin/bash
echo "============================================"
echo "  Redshift Query Cost Analyzer"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Install deps
pip install flask flask-cors --break-system-packages -q 2>/dev/null

echo "Starting backend on http://localhost:5000"
echo "Frontend: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd backend
python app.py
