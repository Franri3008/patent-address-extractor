#!/usr/bin/env python3
"""Pipeline benchmark with isolation modes and OCR concurrency probe.

Modes:
  ocr-probe  — Send N concurrent OCR requests to vLLM (no pipeline).
                Proves whether vLLM benefits from concurrent requests.
  ocr        — Run PDF+OCR pipeline only (dummy LLM sink).
  gemma      — Run full pipeline, focus on LLM metrics.
  full       — Complete pipeline with profiling.

Usage:
    python benchmark.py --mode ocr-probe --patents 10
    python benchmark.py --mode ocr-probe --patents 10 --concurrency 1,2,4,8
    python benchmark.py --mode full --patents 20
    python benchmark.py --mode full --patents 20 --ocr-concurrency 1,2,4,8
    python benchmark.py --mode full --patents 20 --llm-concurrency 1,2,4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("benchmark")


def load_config(path: str = "config.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sample_patents(config: dict, n: int) -> list[dict]:
    """Load first N patent rows from cached BigQuery CSV."""
    import pandas as pd

    out_dir = Path(config["output"]["dir"])
    batch = config.get("batch", {})
    yyyy = str(batch.get("year", "0000"))
    mm = str(batch.get("month", "00")).zfill(2)
    tmpl = config["output"]["filename_template"]
    fname = tmpl.format(yyyy=yyyy, mm=mm)
    raw_csv = out_dir / f"raw_{fname}.csv"

    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run the pipeline first to fetch patent data.")
        sys.exit(1)

    df = pd.read_csv(raw_csv)
    rows = df.head(n).to_dict("records")
    print(f"Loaded {len(rows)} patents from {raw_csv}")
    return rows


# ── OCR Concurrency Probe ──────────────────────────────────────────────────

async def _download_patent_images(patent_rows: list[dict], config: dict) -> list[dict]:
    """Download PDFs and extract page images for a list of patents."""
    from pipeline.pdf_worker import pdf_worker

    patent_q: asyncio.Queue = asyncio.Queue()
    image_q: asyncio.Queue = asyncio.Queue()
    n_pdf = min(config["workers"]["pdf_concurrency"], len(patent_rows))

    for row in patent_rows:
        await patent_q.put(row)
    for _ in range(n_pdf):
        await patent_q.put(None)

    tasks = [
        asyncio.create_task(
            pdf_worker(patent_q, image_q, config, n_sentinels_to_emit=1)
        )
        for _ in range(n_pdf)
    ]
    await asyncio.gather(*tasks)

    items = []
    while not image_q.empty():
        item = image_q.get_nowait()
        if item is not None and item.get("images"):
            items.append(item)
    print(f"Downloaded {len(items)} patents with images")
    return items


async def ocr_probe(patent_items: list[dict], config: dict, concurrency_levels: list[int]) -> list[dict]:
    """Send concurrent OCR requests to vLLM and measure throughput at each level."""
    from models.ocr import get_ocr_model
    from utils.wipo import extract_sections, page_decision

    ocr_model = get_ocr_model(config)
    ocr_model.load()

    results = []
    for conc in concurrency_levels:
        semaphore = asyncio.Semaphore(conc)
        total_pages = 0
        total_patents = 0

        async def process_one(item: dict) -> int:
            nonlocal total_pages
            images = item["images"]
            pages_done = 0
            accumulated_text = ""
            for page_idx, img in enumerate(images, start=1):
                async with semaphore:
                    result = await ocr_model.run_async([img])
                accumulated_text += ("\n" if accumulated_text else "") + result.text
                pages_done += 1
                sections = extract_sections(accumulated_text)
                decision, _ = page_decision(sections)
                if decision == "done":
                    break
            return pages_done

        print(f"\nOCR probe: concurrency={conc}, patents={len(patent_items)}")
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(process_one(item)) for item in patent_items]
        page_counts = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        total_pages = sum(page_counts)
        total_patents = len(patent_items)

        row = {
            "concurrency": conc,
            "patents": total_patents,
            "pages": total_pages,
            "elapsed_s": round(elapsed, 2),
            "patents_per_sec": round(total_patents / elapsed, 3) if elapsed > 0 else 0,
            "pages_per_sec": round(total_pages / elapsed, 3) if elapsed > 0 else 0,
        }
        results.append(row)
        print(f"  -> {row['patents_per_sec']} patents/s, {row['pages_per_sec']} pages/s, {elapsed:.1f}s")

    # Print comparison table
    if len(results) > 1 and results[0]["elapsed_s"] > 0:
        baseline = results[0]["elapsed_s"]
        print(f"\n{'conc':>5} {'patents':>8} {'pages':>6} {'time_s':>7} {'pat/s':>8} {'pg/s':>8} {'speedup':>8}")
        print("-" * 60)
        for r in results:
            speedup = baseline / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
            print(f"{r['concurrency']:>5} {r['patents']:>8} {r['pages']:>6} {r['elapsed_s']:>7} "
                  f"{r['patents_per_sec']:>8} {r['pages_per_sec']:>8} {speedup:>7.1f}x")

    return results


# ── Full Pipeline Benchmark ────────────────────────────────────────────────

async def run_full_benchmark(config: dict, patent_rows: list[dict], label: str = "") -> dict:
    """Run the full pipeline and return profiling aggregates."""
    from main import run_pipeline
    from utils.profiler import PipelineProfiler
    from utils.status_tracker import StatusTracker

    profiler = PipelineProfiler()
    run_id = f"bench_{label}_{int(time.time())}"

    tracker = StatusTracker(
        dashboard_dir=Path("dashboard"),
        run_id=run_id,
        run_mode="batch",
        pipeline_mode=config.get("pipeline_mode", 0),
        total_patents=len(patent_rows),
    )

    t0 = time.perf_counter()
    stats = await run_pipeline(config, patent_rows, run_id, tracker=tracker, profiler=profiler)
    elapsed = time.perf_counter() - t0
    tracker.finish()

    profiler.print_profile_report(elapsed)

    agg = profiler._compute_aggregates(elapsed)
    agg["label"] = label
    agg["config"] = {
        "ocr_concurrency": config["workers"].get("ocr_concurrency", 4),
        "llm_concurrency": config["workers"].get("llm_concurrency", 4),
        "num_ctx": config["llm"].get("num_ctx", 4096),
    }
    return agg


async def sweep_full(config: dict, patent_rows: list[dict],
                     param_name: str, param_values: list[int]) -> list[dict]:
    """Sweep a config parameter and collect results."""
    results = []
    for val in param_values:
        cfg = json.loads(json.dumps(config))  # deep copy
        if param_name == "ocr_concurrency":
            cfg["workers"]["ocr_concurrency"] = val
        elif param_name == "llm_concurrency":
            cfg["workers"]["llm_concurrency"] = val
        elif param_name == "num_ctx":
            cfg["llm"]["num_ctx"] = val

        label = f"{param_name}={val}"
        print(f"\n{'=' * 60}")
        print(f"  Benchmark: {label}")
        print(f"{'=' * 60}")

        agg = await run_full_benchmark(cfg, patent_rows, label=label)
        results.append(agg)

    # Print comparison
    print(f"\n{'=' * 80}")
    print(f"  SWEEP RESULTS: {param_name}")
    print(f"{'=' * 80}")
    print(f"{'config':>20} {'pat/s':>8} {'avg_s':>7} {'p50_s':>7} {'p95_s':>7}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:>20} {r['patents_per_sec']:>8.3f} "
              f"{r['avg_sec_per_patent']:>7.3f} "
              f"{r['total_wall'].get('p50', 0):>7.3f} "
              f"{r['total_wall'].get('p95', 0):>7.3f}")
    print(f"{'=' * 80}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--mode", required=True,
                        choices=["ocr-probe", "ocr", "gemma", "full"],
                        help="Benchmark mode")
    parser.add_argument("--patents", type=int, default=20, help="Number of patents to use")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--concurrency", default=None,
                        help="Comma-separated concurrency levels for ocr-probe")
    parser.add_argument("--ocr-concurrency", default=None,
                        help="Comma-separated OCR concurrency levels for full sweep")
    parser.add_argument("--llm-concurrency", default=None,
                        help="Comma-separated LLM concurrency levels for full sweep")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    config["run_mode"] = "batch"
    patent_rows = load_sample_patents(config, args.patents)

    if args.mode == "ocr-probe":
        levels = [int(x) for x in (args.concurrency or "1,2,4,8").split(",")]
        # First download images
        print("Phase 1: Downloading patent PDFs and extracting images...")
        items = asyncio.run(_download_patent_images(patent_rows, config))
        if not items:
            print("No images to benchmark. Check network/PDF availability.")
            sys.exit(1)
        print(f"\nPhase 2: OCR concurrency probe with {len(items)} patents...")
        results = asyncio.run(ocr_probe(items, config, levels))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    elif args.mode == "full":
        if args.ocr_concurrency:
            levels = [int(x) for x in args.ocr_concurrency.split(",")]
            results = asyncio.run(sweep_full(config, patent_rows, "ocr_concurrency", levels))
        elif args.llm_concurrency:
            levels = [int(x) for x in args.llm_concurrency.split(",")]
            results = asyncio.run(sweep_full(config, patent_rows, "llm_concurrency", levels))
        else:
            result = asyncio.run(run_full_benchmark(config, patent_rows, label="default"))
            results = [result]

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)

    elif args.mode in ("ocr", "gemma"):
        # For now, run full pipeline with profiling — the profiler report
        # separates per-stage metrics so you can read OCR or Gemma in isolation.
        # For true isolation (one server stopped), stop the other server manually
        # and run --mode full.
        print(f"Running full pipeline (read {args.mode} metrics from the profile report).")
        print("For true isolation, stop the other server and use --mode full.")
        result = asyncio.run(run_full_benchmark(config, patent_rows, label=args.mode))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
