#!/bin/bash
# Restart script for Vast.ai server
# Run from anywhere: bash /workspace/GP/GP/dashboard/restart.sh

cd /workspace/GP/GP
git pull
cd dashboard/frontend
npm run build
cd ../backend
pkill -f uvicorn
sleep 1
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
echo "Server restarted. Check nohup.out for logs."
