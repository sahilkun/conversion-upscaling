#!/bin/bash
# Fish Speech S2 Pro - RunPod One-Time Setup
# Run this ONCE after pod starts
set -e

echo "=== Fish Speech S2 Pro Setup ==="
echo "Step 1: Downloading S2 Pro model (~11 GB)..."

# Download model if not already present
if [ ! -f "/app/checkpoints/s2-pro/codec.pth" ] && [ ! -f "/workspace/s2-pro/codec.pth" ]; then
    pip install -U huggingface_hub 2>/dev/null
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/s2-pro', local_dir='/app/checkpoints/s2-pro')
print('Model downloaded!')
"
else
    echo "Model already downloaded, skipping."
fi

echo ""
echo "Step 2: Installing extra dependencies..."
pip install demucs pydub 2>&1 | tail -3

echo ""
echo "Step 3: Starting API server..."
# Kill any existing server
pkill -f "api_server" 2>/dev/null || true
sleep 2

# Start the Fish Speech API server
nohup python -m fish_speech.tools.api_server \
    --llama-checkpoint-path /app/checkpoints/s2-pro \
    --decoder-checkpoint-path /app/checkpoints/s2-pro/codec.pth \
    --listen 0.0.0.0:8080 \
    --half \
    > /workspace/server.log 2>&1 &

echo "Server PID: $!"
echo ""
echo "Step 4: Waiting for model to load (60-90s)..."

# Wait for server to be ready
for i in $(seq 1 30); do
    sleep 3
    if curl -s http://127.0.0.1:8080/docs > /dev/null 2>&1; then
        echo ""
        echo "=== Server READY! ==="
        echo "API: http://127.0.0.1:8080"
        echo "Docs: http://127.0.0.1:8080/docs"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
        exit 0
    fi
    printf "."
done

echo ""
echo "Server may still be loading. Check: tail -20 /workspace/server.log"
