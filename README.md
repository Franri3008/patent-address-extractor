# patent-address-extractor

Pipeline for extracting inventor and applicant addresses from WO (PCT) patents. It i) queries Google BigQuery for patent metadata, ii) downloads patent PDFs, iii) runs OCR, iv) and uses an LLM to extract structured address data.

## Features

- Two modes: `individual` (verbose, for testing) and `batch` (production, BQ has ~30k patents per month so it expects that)
- A sort of reliable heuristic for page fetching (sometimes addresses are in pages beyond the first one)
- 'Plug-and-play' OCR and LLM models; add a new model by creating one file
- Concurrent pipeline with configurable worker pools per stage (this is being tested)
- Dashboard-ready metadata (dashboard it for the future)

## Requirements

- Python 3.11+
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud` CLI) for BigQuery auth
- Poppler (for `pdf2image`):
  - Linux: `apt-get install poppler-utils`
  - Mac: `brew install poppler`

## Setup

```bash
pip install -r requirements.txt

gcloud auth application-default login
cp config.example.json config.json
# Edit config.json: set project_id, year/month, OCR model, LLM provider
```

## OCR: PaddleOCR-VL (default)

A 0.9B vision-language model from PaddlePaddle. Uses the HuggingFace transformers API — no `paddlepaddle` package required. Weights (~1.9 GB) download automatically on first run.

```bash
pip install torch torchvision transformers
```

Set in `config.json`:
```json
"ocr": { "model": "paddle_ocr", "device": "auto" }
```

Device selection: `"auto"` picks CUDA → MPS (Apple Silicon) → CPU.

> **Apple Silicon (MPS) note:** The model runs with `attn_implementation="eager"` on MPS to avoid an SDPA kernel crash. Expect ~75s/page. For production throughput, a Linux machine with CUDA is recommended.

## LLM Setup

Local: LM Studio or Ollama (OpenAI-compatible endpoint)
```bash
# LM Studio: start server, then set in config.json:
# "provider": "openai", "model": "gemma-3-4b-it", "base_url": "http://127.0.0.1:1234/v1"

# Ollama:
ollama pull gemma3:27b
# "provider": "ollama", "model": "gemma3:27b"
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

## Usage

```bash
# Individual patent
python main.py  # uses config.json run_mode = "individual"

# BigQuery only
python pipeline/bq_fetcher.py --year 2025 --month 1 --limit 10
python pipeline/bq_fetcher.py --patent WO2025086418
```

## Configuration (`config.json`)

| Key                          | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `run_mode`                   | `"individual"` or `"batch"`                         |
| `individual.patent_id`       | Patent ID for individual mode                       |
| `batch.year` / `batch.month` | Target month for batch mode                         |
| `batch.limit`                | Row limit (null = all)                              |
| `bigquery.project_id`        | Your GCP project                                    |
| `pdf.max_pages`              | Max pages to extract (default: 3)                   |
| `pdf.dpi`                    | Resolution for OCR images (default: 150)            |
| `ocr.model`                  | `"paddle_ocr"` (default)                            |
| `ocr.device`                 | `"auto"`, `"cuda"`, `"mps"`, `"cpu"`                |
| `llm.provider`               | `"openai"`, `"ollama"`, `"anthropic"`, `"google"`   |
| `llm.model`                  | Model name (e.g. `"gemma-3-4b-it"`, `"gpt-4o-mini"`) |
| `llm.base_url`               | Override API endpoint (for LM Studio / local Ollama) |
| `workers.pdf_concurrency`    | Parallel PDF download workers (I/O-bound, try 8–16) |
| `workers.ocr_workers`        | OCR thread pool size (usually 1 per GPU)            |
| `workers.llm_concurrency`    | Parallel LLM callers (1 for local, 4–8 for API)     |
| `workers.queue_max_size`     | Bounded queue size between stages (backpressure)    |

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

Bounded queues between each stage provide automatic backpressure: if OCR is slower than PDF download, the image queue fills and download workers pause automatically — no memory overflow.
