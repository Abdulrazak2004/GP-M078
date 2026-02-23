#!/bin/bash
# ─── Vast.ai Deployment Script ───────────────────────────────────────────────
# Run this ONCE after SSH-ing into your Vast.ai instance.
# Everything runs on a single port (8000).
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo "=========================================="
echo "  Corrosion Intelligence Platform Deploy"
echo "=========================================="

# 1. Clone the repo
echo "[1/5] Cloning repository..."
if [ ! -d "GP" ]; then
    git clone https://github.com/Abdulrazak2004/GP-M078.git GP
fi
cd GP

# 2. Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q fastapi uvicorn[standard] torch numpy pandas scikit-learn joblib

# 3. Install Node.js (if not present) + build frontend
echo "[3/5] Building frontend..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
cd dashboard/frontend
npm install
npm run build
cd ../..

# 4. Start the server
echo "[4/5] Starting server on port 8000..."
echo ""
echo "=========================================="
echo "  Dashboard will be available at:"
echo "  http://<YOUR_VAST_IP>:8000"
echo "=========================================="
echo ""

cd dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
