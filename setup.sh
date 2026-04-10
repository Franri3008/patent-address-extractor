#!/usr/bin/env bash
# setup.sh — one-shot environment bootstrap for patent-address-extractor
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Detect OS ──────────────────────────────────────────────────────────────
OS="$(uname -s)"
info "Detected OS: $OS"

# ── 2. Poppler ────────────────────────────────────────────────────────────────
info "Checking poppler..."
if command -v pdftoppm &>/dev/null; then
    ok "poppler already installed"
elif [[ "$OS" == "Darwin" ]]; then
    command -v brew &>/dev/null || die "Homebrew not found. Install it first: https://brew.sh"
    brew install poppler
elif [[ "$OS" == "Linux" ]]; then
    sudo apt-get update -qq && sudo apt-get install -y poppler-utils
else
    warn "Unknown OS — install poppler manually and re-run."
fi

# ── 3. Ollama ─────────────────────────────────────────────────────────────────
info "Checking ollama..."
if ! command -v ollama &>/dev/null; then
    info "Installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    ok "ollama already installed"
fi

# Start ollama as a background daemon if not already running
# Tuning: OLLAMA_NUM_PARALLEL controls concurrent inference slots (each needs its own KV cache).
#         OLLAMA_FLASH_ATTENTION enables Flash Attention 2 (faster, less memory on H100).
#         Override with env vars: OLLAMA_NUM_PARALLEL=6 ./setup.sh
if ! pgrep -x ollama &>/dev/null; then
    info "Starting ollama daemon..."
    OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4} \
    OLLAMA_FLASH_ATTENTION=1 \
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
    ok "ollama daemon started (NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}, FLASH_ATTENTION=1, logs → /tmp/ollama.log)"
else
    ok "ollama daemon already running"
fi

# ── 4. Python virtualenv + dependencies ──────────────────────────────────────
info "Setting up Python virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
ok "Python dependencies installed"

# ── 5. Config ─────────────────────────────────────────────────────────────────
if [[ ! -f config.json ]]; then
    if [[ -f config.example.json ]]; then
        cp config.example.json config.json
        warn "config.json created from config.example.json — review it before running."
    else
        warn "No config.json found. Copy and edit one before running main.py."
    fi
else
    ok "config.json already present"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
ok "Setup complete. Activate the venv with:  source .venv/bin/activate"
echo ""
echo "  Run the pipeline:      python main.py"
echo "  Run the review server: python main.py --review"
echo "  Run tests:             python main.py --test"
echo ""
echo "vLLM servers (start separately, each on its own port):"
echo ""
echo "  # OCR server (PaddleOCR-VL, port 8000):"
echo "  vllm serve PaddlePaddle/PaddleOCR-VL \\"
echo "      --trust-remote-code --served-model-name PaddleOCR-VL-0.9B \\"
echo "      --port 8000 --gpu-memory-utilization 0.35 \\"
echo "      --max-num-batched-tokens 32768 --max-num-seqs 8 \\"
echo "      --no-enable-prefix-caching --mm-processor-cache-gb 2"
echo ""
echo "  # LLM server (Gemma for address extraction, port 8001):"
echo "  vllm serve google/gemma-3-4b-it \\"
echo "      --port 8001 --served-model-name gemma \\"
echo "      --gpu-memory-utilization 0.55 --max-num-seqs 32 \\"
echo "      --max-model-len 4096 --dtype auto \\"
echo "      --enable-prefix-caching --disable-log-requests"
