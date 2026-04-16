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

# ── 4b. vLLM + transformers for Gemma4 ───────────────────────────────────────
# The stable pip wheel for vllm pins `transformers<5`, but Gemma4 weights
# declare `model_type: "gemma4"`, which is only handled in transformers v5+.
# We install from nightly/source and deliberately ignore pip's version-pin
# warnings (the pins are stale; runtime imports resolve correctly).
info "Installing vllm nightly + transformers v5 for Gemma4..."
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install --upgrade --force-reinstall --no-deps \
    git+https://github.com/huggingface/transformers.git
pip install --upgrade 'huggingface_hub>=1.0'   # transformers v5 needs hf_hub 1.x
pip install --no-deps 'compressed-tensors==0.14.0.1'  # match vllm nightly's pin
ok "vllm nightly + transformers v5 installed (pip pin warnings are expected)"

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
echo "  # Gemma 4 support (vllm nightly + transformers v5) is installed by"
echo "  # this script above — pip's resolver will print version-pin warnings"
echo "  # for vllm/compressed-tensors vs transformers v5; they are expected"
echo "  # and do not block runtime imports."
echo ""
echo "  # Keep VLLM_USE_DEEP_GEMM=0 unless you have the deep_gemm kernels"
echo "  # installed — otherwise FP8 warmup will crash with 'DeepGEMM backend is"
echo "  # not available or outdated'."
echo ""
echo "  # google/gemma-4-E2B-it is a gated repo — run 'huggingface-cli login'"
echo "  # (or export HF_TOKEN=...) with an account that has accepted the"
echo "  # Gemma license before starting the LLM server."
echo ""
echo "  # OCR server (PaddleOCR-VL, port 8000):"
echo "  VLLM_USE_DEEP_GEMM=0 vllm serve PaddlePaddle/PaddleOCR-VL \\"
echo "      --trust-remote-code --served-model-name PaddleOCR-VL-0.9B \\"
echo "      --port 8000 --gpu-memory-utilization 0.35 \\"
echo "      --max-num-batched-tokens 32768 --max-num-seqs 16 \\"
echo "      --no-enable-prefix-caching --mm-processor-cache-gb 2"
echo ""
echo "  # LLM server (Gemma4 E2B for address extraction, port 8001):"
echo "  VLLM_USE_DEEP_GEMM=0 vllm serve google/gemma-4-E2B-it \\"
echo "      --port 8001 --served-model-name gemma4 \\"
echo "      --gpu-memory-utilization 0.55 --max-num-seqs 32 \\"
echo "      --max-model-len 4096 \\"
echo "      --limit-mm-per-prompt '{\"image\": 0, \"audio\": 0}' \\"
echo "      --enable-prefix-caching \\"
echo "      --quantization fp8"
