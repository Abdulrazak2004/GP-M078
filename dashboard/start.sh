#!/bin/bash
# Start both backend and frontend for the Corrosion Intelligence Dashboard

set -e

echo "╔══════════════════════════════════════════╗"
echo "║  CORROSION INTELLIGENCE PLATFORM         ║"
echo "║  Starting Dashboard...                   ║"
echo "╚══════════════════════════════════════════╝"

# Start backend
echo ""
echo "→ Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")/backend"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
echo "→ Starting Vite frontend on port 5173..."
cd "$(dirname "$0")/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Dashboard will be ready at: http://localhost:5173"
echo "(Backend data loading takes ~30-60 seconds)"
echo ""
echo "Press Ctrl+C to stop both servers."

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
