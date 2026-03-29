#!/bin/bash
# Fish Speech S2 Pro - RunPod Setup Script
# Run this once after pod starts

set -e
echo "=== Setting up Fish Speech S2 Pro ==="

# Install Fish Speech
cd /workspace
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e . 2>&1 | tail -5

# Download S2 Pro model (~15 GB)
echo "=== Downloading S2 Pro model ==="
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro

echo "=== Setup complete! ==="
echo "Fish Speech S2 Pro ready at /workspace/fish-speech"
nvidia-smi
