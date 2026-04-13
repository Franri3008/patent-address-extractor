# patent-address-extractor

Pipeline for extracting inventor and applicant addresses from WO (PCT) patents. It i) queries Google BigQuery for patent metadata, ii) downloads patent PDFs, iii) runs OCR, iv) and uses an LLM to extract structured address data.

## Features

- Two modes: `individual` (verbose, for testing) and `batch` (production, BQ has ~30k patents per month so it expects that)
- A sort of reliable heuristic for page fetching (sometimes addresses are in pages beyond the first one)
- 'Plug-and-play' OCR and LLM models; add a new model by creating one file
- Concurrent pipeline with configurable worker pools per stage (this is being tested)
- Dashboard-ready metadata (dashboard is for the future)

## Requirements

- Python 3.11+
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud` CLI) for BigQuery auth
- Poppler (for `pdf2image`):
  - Linux: `apt-get update && apt-get install -y poppler-utils`
  - Mac: `brew install poppler`

## Setup

```bash
pip install -r requirements.txt

gcloud auth application-default login
cp config.example.json config.json
# Edit config.json: set project_id, year/month, OCR model, LLM provider
```

## OCR: PaddleOCR-VL

A 0.9B vision-language model from PaddlePaddle. Two backends are available:

### Option A — local (transformers)

Loads the model in-process via HuggingFace transformers. No `paddlepaddle` package required. Weights (~1.9 GB) download automatically on first run.

```bash
pip install torch torchvision transformers
```

```json
"ocr": { "model": "paddle_ocr", "device": "auto" }
```

Device selection: `"auto"` picks CUDA → MPS (Apple Silicon) → CPU.

> **Apple Silicon (MPS) note:** The model runs with `attn_implementation="eager"` on MPS to avoid an SDPA kernel crash. Expect ~75s/page. For production throughput, a Linux machine with CUDA is recommended.

### Option B — vLLM server

Runs the model on a vLLM server; the pipeline only needs the `openai` client. No GPU required on the pipeline host.

1. Start the server (on a GPU machine):
   ```bash
   pip install vllm
   
   VLLM_USE_DEEP_GEMM=0 vllm serve PaddlePaddle/PaddleOCR-VL \
   --trust-remote-code --served-model-name PaddleOCR-VL-0.9B \
   --port 8000 --gpu-memory-utilization 0.40 \
   --max-num-batched-tokens 32768 --max-num-seqs 16 \
   --no-enable-prefix-caching --mm-processor-cache-gb 2
   ```

2. Set in `config.json`:
   ```json
   "ocr": {
     "model": "paddle_ocr_vllm",
     "vllm_base_url": "http://localhost:8000/v1",
     "vllm_model": "PaddlePaddle/PaddleOCR-VL"
   }
   ```

## LLM Setup

### Mode 0 (default) — OCR + LLM

Local: LM Studio or Ollama (OpenAI-compatible endpoint)
```bash
# LM Studio: start server, then set in config.json:
# "provider": "openai", "model": "gemma-3-4b-it", "base_url": "http://127.0.0.1:1234/v1"

# Ollama — install:
#   Mac:   brew install ollama
#   Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b
# "provider": "ollama", "model": "gemma4:e2b"
```

API providers:
```bash
# OpenAI
export OPENAI_API_KEY=sk-...
# "provider": "openai", "model": "gpt-4o-mini"

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
export GOOGLE_API_KEY=...
```

API keys can also be set directly in `config.json` under `llm.api_key` instead of using env vars.

### Mode 1 — Vision LLM (no OCR)

Skips the OCR stage entirely. A vision-capable LLM reads page images directly and extracts addresses in one shot. Pages are processed one by one (up to 3) and processing stops as soon as the inventor section `(72)` is detected.

1. Install Ollama if not already installed:
   ```bash
   # Mac
   brew install ollama
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   # Windows: download installer from https://ollama.com/download
   ```

2. Pull a vision-capable model:
   ```bash
   ollama pull gemma4:e2b
   # or any other vision model
   ```

3. Ollama starts automatically when you run a model, but you can also start the server explicitly:
   ```bash
   ollama serve
   ```

4. Set in `config.json`:
   ```json
   "pipeline_mode": 1,
   "vision_llm": {
     "provider": "ollama",
     "model": "gemma4:e2b",
     "temperature": 0.0,
     "max_pages": 3,
     "max_retries": 2
   }
   ```

5. Or override at runtime:
   ```bash
   python main.py --pipeline-mode 1
   ```

## Usage

```bash
# Individual patent (uses config.json run_mode)
python main.py

# Override run mode and pipeline mode at runtime
python main.py --mode individual --patent WO2025086418
python main.py --pipeline-mode 1   # use vision LLM

# BigQuery only
python pipeline/bq_fetcher.py --year 2025 --month 1 --limit 10
python pipeline/bq_fetcher.py --patent WO2025086418
```

## Configuration (`config.json`)

| Key                          | Description                                                      |
| ---------------------------- | ---------------------------------------------------------------- |
| `run_mode`                   | `"individual"` or `"batch"`                                      |
| `individual.patent_id`       | Patent ID for individual mode                                    |
| `batch.year` / `batch.month` | Target month for batch mode                                      |
| `batch.limit`                | Row limit (null = all)                                           |
| `bigquery.project_id`        | Your GCP project                                                 |
| `pdf.max_pages`              | Max pages to extract (default: 3)                                |
| `pdf.dpi`                    | Resolution for OCR images (default: 150)                         |
| `ocr.model`                  | `"paddle_ocr"`, `"paddle_ocr_vllm"`, `"dots_ocr"`                |
| `ocr.device`                 | `"auto"`, `"cuda"`, `"mps"`, `"cpu"` (local only)                |
| `ocr.vllm_base_url`          | vLLM server URL (default: `http://localhost:8000/v1`)            |
| `ocr.vllm_model`             | Model name served by vLLM (default: `PaddlePaddle/PaddleOCR-VL`) |
| `pipeline_mode`              | `0` = OCR + LLM (default), `1` = vision LLM only                 |
| `llm.provider`               | `"openai"`, `"ollama"`, `"anthropic"`, `"google"`                |
| `llm.model`                  | Model name (e.g. `"gemma-3-4b-it"`, `"gpt-4o-mini"`)             |
| `llm.api_key`                | API key inline (alternative to env var)                          |
| `llm.base_url`               | Override API endpoint (for LM Studio / local Ollama)             |
| `vision_llm.provider`        | Vision LLM provider (mode 1 only); `"ollama"`                    |
| `vision_llm.model`           | Vision model name (e.g. `"gemma4:12b"`)                          |
| `vision_llm.max_pages`       | Max pages to try before giving up (default: 3)                   |
| `workers.pdf_concurrency`    | Parallel PDF download workers (I/O-bound, try 8–16)              |
| `workers.ocr_workers`        | OCR thread pool size (usually 1 per GPU)                         |
| `workers.llm_concurrency`    | Parallel LLM callers (1 for local, 4–8 for API)                  |
| `workers.queue_max_size`     | Bounded queue size between stages (backpressure)                 |

## Output

```
output/
├── raw_patents_2025_01.csv          # Raw BigQuery data
├── patents_2025_01.csv              # Final output with addresses
├── metadata_patents_2025_01.jsonl   # Per-patent metadata (for dashboard)
├── report_patents_2025_01.json      # Batch summary report
└── individual/
    └── WO2025086418/
        ├── result.csv
        ├── metadata.json            # Full individual metadata (pretty)
        └── page_1_thumb.jpg         # Page thumbnail(s)
```

### Output CSV columns

`publication_number`, `title_text`, `application_number`, `publication_date`, `inventor_names`, `assignee_names`, `cpc_codes`, `google_patents_url`, **`inventors_with_address`** (JSON), **`applicants_with_address`** (JSON), **`agents_with_address`** (JSON), `addresses_found`, `pdf_type`, `pages_used`, `page_reason`, `ocr_model`, `llm_provider`, `ocr_elapsed_s`, `llm_elapsed_s`, `llm_cost_usd`, `error`

## Accuracy Improvements

Several optional modules improve extraction accuracy on noisy, scanned, and two-column patents. All are disabled by default and enabled via `config.json`.

### Structured Output Enforcement

All LLM backends now pass a JSON schema to the API when the provider supports it natively. This eliminates JSON parse failures and prevents the model from inventing extra fields or returning malformed responses.

| Provider  | Enforcement method                                |
| --------- | ------------------------------------------------- |
| Ollama    | `format=` parameter with JSON schema              |
| OpenAI    | `response_format.json_schema` with `strict: true` |
| Google    | `response_schema` in `GenerationConfig`           |
| Anthropic | Prompt-based (no native schema mode)              |

No config change required — this is always active.

### Known-Entity Hints from BigQuery

Before calling the LLM, the pipeline injects the known applicant and inventor names from BigQuery into the prompt. This helps the LLM find all expected entities even when OCR garbles names (e.g. in two-column layouts). The LLM is instructed to locate each known name in the text and attribute the correct address to it.

```json
"llm": {
  "template_vars": {}  // automatically populated from BigQuery row
}
```

No config change required — names are always injected when available.

### Post-Validation (3C)

Deterministic checks that run after LLM extraction with no extra model calls. Issues are stored as `validation_warnings` in the metadata JSONL and used to trigger vision verification when enabled.

Checks performed:
- **Country codes** — validates `(XX)` codes in extracted addresses against ISO 3166-1 alpha-2
- **Entity completeness** — flags known BigQuery applicants/inventors missing from the extraction (fuzzy match)
- **Section consistency** — flags mismatches between OCR-detected sections and LLM-reported sections

```json
"llm": {
  "post_validation": {
    "enabled": true
  }
}
```

### Two-Column Layout Detection (4A+4B)

WIPO first pages have a two-column layout in the top ~60–70% of the page (left column: bibliographic data with sections 71, 72, 74; right column: classification codes, etc.), followed by a horizontal separator line, then full-width content below (title, abstract, drawings). OCR naively reads across both columns, mixing the text.

When enabled, the pipeline:
1. Detects the horizontal separator line and vertical column gap using projection profiles (no GPU required)
2. On page 1 only, crops just the left column above the separator before running OCR
3. Falls back to full-page OCR if no two-column layout is detected (confidence below threshold)

```json
"column_detection": {
  "enabled": true,
  "confidence_threshold": 0.7
}
```

Requires `numpy` (already a transitive dependency via PyTorch).

### Vision LLM Verification Fallback (4D)

When post-validation flags issues (missing known entities, unknown country codes, section mismatches), the pipeline automatically re-extracts addresses by sending the original page image to a vision-capable LLM. The vision result replaces the OCR+LLM result if the extraction succeeds.

Recommended model: `gemma4:e2b` (or any Ollama vision model). The verification model is configured separately from the main pipeline model.

```json
"verification": {
  "enabled": true,
  "vision_llm": {
    "provider": "ollama",
    "model": "gemma4:e2b",
    "temperature": 0.0
  }
}
```

**Recommended stack for maximum accuracy:**
```json
"llm": { "post_validation": { "enabled": true } },
"column_detection": { "enabled": true, "confidence_threshold": 0.7 },
"verification": { "enabled": true, "vision_llm": { "provider": "ollama", "model": "gemma4:e2b", "temperature": 0.0 } }
```

---

## Adding a New OCR Model

1. Create `models/ocr/my_model.py` subclassing `OCRModel`
2. Implement `load()`, `run(images)`, and `model_name`
3. Add one entry to `models/ocr/__init__.py`:
   ```python
   if key == "my_model":
       from models.ocr.my_model import MyModel
       return MyModel(config)
   ```

## Adding a New LLM Provider

Same pattern — subclass `LLMModel`, implement `extract_addresses()`, register in `models/llm/__init__.py`.

## Pipeline Architecture

**Mode 0 (default):**
```
[BigQuery Fetch] → raw CSV
       ↓
[Patent Queue] → N PDF Workers (async + thread pool)
       ↓
[Image Queue]  → OCR Coordinator (ThreadPoolExecutor, 1/GPU)
                    — WIPO section-number heuristic per page
       ↓
[Text Queue]   → N LLM Workers (async)
       ↓
[Result Queue] → Output Writer (single, append-safe)
                    → CSV + JSONL + delete temp PDF
```

**Mode 1 (vision LLM):**
```
[BigQuery Fetch] → raw CSV
       ↓
[Patent Queue] → N PDF Workers (async + thread pool)
       ↓
[Image Queue]  → N Vision LLM Workers (async, page-by-page)
                    — stops when (72) inventor section found, max 3 pages
       ↓
[Result Queue] → Output Writer (single, append-safe)
                    → CSV + JSONL + delete temp PDF
```

Bounded queues between each stage provide automatic backpressure: if OCR is slower than PDF download, the image queue fills and download workers pause automatically — no memory overflow.

---

## Quick Reference

### Setup

```bash
pip install -r requirements.txt                                          # install dependencies
gcloud auth application-default login                                    # BigQuery auth
cp config.example.json config.json                                       # create config
```

### Running the pipeline

```bash
python main.py                                                           # run using config.json settings
python main.py --mode individual --patent WO2025086418                   # single patent
python main.py --mode batch                                              # full batch (year/month from config)
python main.py --pipeline-mode 1                                         # vision LLM (no OCR)
python main.py --pipeline-mode 0                                         # OCR + LLM (default)
```

### Testing & review

```bash
python main.py --test                                                    # regression test against ground truth
python main.py --review                                                  # start review dashboard (port 8765)
```

### BigQuery fetch only

```bash
python pipeline/bq_fetcher.py --year 2025 --month 1 --limit 10          # fetch 10 patents
python pipeline/bq_fetcher.py --patent WO2025086418                      # single patent metadata
```

### vLLM servers (run on GPU machine, each in its own terminal)

```bash
# OCR server (PaddleOCR-VL, port 8000):
VLLM_USE_DEEP_GEMM=0 vllm serve PaddlePaddle/PaddleOCR-VL \
  --trust-remote-code --served-model-name PaddleOCR-VL-0.9B \
  --port 8000 --gpu-memory-utilization 0.40 \
  --max-num-batched-tokens 32768 --max-num-seqs 16 \
  --no-enable-prefix-caching --mm-processor-cache-gb 2

# LLM server (Gemma4 E2B, port 8001):
VLLM_USE_DEEP_GEMM=0 vllm serve google/gemma-4-E2B-it \
  --port 8001 --served-model-name gemma4 \
  --gpu-memory-utilization 0.50 --max-num-seqs 32 \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --enable-prefix-caching
```

### Ollama (local LLM)

```bash
ollama pull gemma4:e2b                                                   # pull default model
ollama pull gemma4:12b                                                   # pull larger model
ollama serve                                                             # start server (auto-starts on first use)
```

### Enable accuracy improvements (in config.json)

```bash
# Post-validation (deterministic checks, no extra cost)
# Set: "llm": { "post_validation": { "enabled": true } }

# Two-column layout splitting (fixes OCR garbling on WIPO first pages)
# Set: "column_detection": { "enabled": true, "confidence_threshold": 0.7 }

# Vision LLM verification fallback (re-extracts via image when validation fails)
# Set: "verification": { "enabled": true, "vision_llm": { "provider": "ollama", "model": "gemma4:e2b", "temperature": 0.0 } }
```
