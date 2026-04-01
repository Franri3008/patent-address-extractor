# patent-address-extractor

A modular, production-grade pipeline for extracting **inventor and applicant addresses** from WO (PCT) patents. It queries Google BigQuery for patent metadata, downloads patent PDFs, runs OCR, and uses an LLM to extract structured address data — all in a concurrent, resource-allocated async pipeline.

## Features

- **Two modes**: `individual` (verbose, for testing) and `batch` (production, ~30k patents/month)
- **Adaptive page fetching** using WIPO sequential section-number heuristic — fetches only the pages needed
- **Plug-and-play OCR and LLM models**: add a new model by creating one file
- **Concurrent pipeline** with configurable worker pools per stage
- **Dashboard-ready metadata** — per-patent JSONL with timing, token counts, page thumbnails

## Requirements

- Python 3.11+
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud` CLI) for BigQuery auth
- [Ollama](https://ollama.com/) for local LLM inference (or an API key for cloud providers)
- Poppler (for `pdf2image`):
  - Linux: `apt-get install poppler-utils`
  - Mac: `brew install poppler`

## Setup

```bash
git clone https://github.com/yourname/patent-address-extractor
cd patent-address-extractor
pip install -r requirements.txt

# BigQuery authentication (Application Default Credentials)
gcloud auth application-default login

# Copy and fill in config
cp config.example.json config.json
# Edit config.json: set project_id, year/month, OCR model, LLM provider
```

## OCR: dots.ocr-1.5

Install via ModelScope (first run downloads weights automatically):

```bash
pip install modelscope torch torchvision transformers
```

Or follow setup instructions at:
https://github.com/rednote-hilab/dots.ocr

> **After installing**: run a quick test to confirm the model loads correctly.
> If the ModelScope pipeline interface differs from expected, update `_run_inference()`
> in `models/ocr/dots_ocr.py` accordingly.

## LLM Setup

**Local (Ollama — recommended):**
```bash
# Install Ollama: https://ollama.com/
ollama pull gemma3:27b
# Set in config.json: "provider": "ollama", "model": "gemma3:27b"
```

**API providers:**
```bash
# OpenAI
export OPENAI_API_KEY=sk-...
# Set in config.json: "provider": "openai", "model": "gpt-4o-mini"

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
export GOOGLE_API_KEY=...
```

## Usage

```bash
# Individual patent (verbose, keeps files, full metadata)
python main.py --mode individual --patent WO2025086418

# Full batch (month)
python main.py --mode batch

# BigQuery only (standalone, useful for testing connectivity)
python pipeline/bq_fetcher.py --year 2025 --month 1 --limit 10
python pipeline/bq_fetcher.py --patent WO2025086418
```

## Configuration (`config.json`)

| Key | Description |
|---|---|
| `run_mode` | `"individual"` or `"batch"` |
| `batch.year` / `batch.month` | Target month for batch mode |
| `batch.limit` | Row limit (null = all) |
| `bigquery.project_id` | Your GCP project |
| `pdf.max_pages` | Max pages to extract (default: 3) |
| `pdf.dpi` | Resolution for OCR images (default: 150) |
| `ocr.model` | OCR model key (currently: `"dots_ocr"`) |
| `ocr.device` | `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `llm.provider` | `"ollama"`, `"openai"`, `"anthropic"`, `"google"` |
| `llm.model` | Model name (e.g. `"gemma3:27b"`, `"gpt-4o-mini"`) |
| `workers.pdf_concurrency` | Parallel PDF download workers (I/O-bound, try 8–16) |
| `workers.ocr_workers` | OCR thread pool size (usually 1 per GPU) |
| `workers.llm_concurrency` | Parallel LLM callers (1 for local, 4–8 for API) |
| `workers.queue_max_size` | Bounded queue size between stages (backpressure) |

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
