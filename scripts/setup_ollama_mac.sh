#!/bin/bash
# Setup Ollama + best Qwen model for M2 Max 32GB. Run on your Mac.
set -e

echo "=== Ollama + Qwen for M2 Max 32GB ==="

# 1. Install Ollama (pick one)
if command -v ollama &>/dev/null; then
  echo "Ollama already installed."
else
  echo "Install Ollama first:"
  echo "  Option A: Download from https://ollama.com/download (drag Ollama to Applications)"
  echo "  Option B: brew install ollama   # if you have Homebrew"
  echo ""
  read -p "After installing Ollama, press Enter to continue..."
fi

if ! command -v ollama &>/dev/null; then
  echo "Ollama not found in PATH. Install it and run this script again."
  exit 1
fi

# 2. Start Ollama in background if not already running
if ! curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &
  sleep 3
fi

# 3. Pull best model for 32GB: Qwen 2.5 32B (fits in ~20GB quantized) or 14B for extra headroom
# For coding, qwen2.5-coder is also excellent; 32B fits your RAM.
MODEL="${1:-qwen2.5:32b}"
echo "Pulling model: $MODEL (this may take several minutes)..."
ollama pull "$MODEL"

echo ""
echo "Done. Installed model: $MODEL"
echo "In Cursor: Settings -> Models -> enable Ollama -> select $MODEL"
echo ""
echo "Alternative coding model: ollama pull qwen2.5-coder:32b"
