#!/usr/bin/env python3
"""Standalone Gemma LLM throughput benchmark via Ollama.

Does NOT require real patents, vLLM, or BigQuery — just a running Ollama server
with the target model pulled.

Tests concurrency levels and num_ctx values to find optimal Ollama configuration.

Usage:
    python benchmark_llm.py
    python benchmark_llm.py --concurrency 1,2,4,6 --num-ctx 2048,4096,8192 --iterations 10
    python benchmark_llm.py --model gemma4:e2b --warmup
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama

# ── Realistic patent OCR sample (~1500 chars, ~400 tokens) ──────────────────
SAMPLE_OCR = """(71) Applicant: ACME CORPORATION; 123 Innovation Drive,
Suite 400, San Jose, California 95134 (US).

(72) Inventors: SMITH, John Robert; 456 Oak Lane, Palo Alto,
California 94301 (US). TANAKA, Yuki; 7-2-1 Nishi-Shinjuku,
Shinjuku-ku, Tokyo 160-0023 (JP). MUELLER, Hans Friedrich;
Maximilianstrasse 42, 80539 Munchen (DE).

(74) Agent: BAKER & ASSOCIATES LLP; 789 Legal Plaza,
Washington, D.C. 20005 (US)."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "inventors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "applicants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "sections_detected": {"type": "array", "items": {"type": "string"}},
        "found": {"type": "boolean"},
    },
    "required": ["inventors", "applicants", "agents", "sections_detected", "found"],
    "additionalProperties": False,
}

# Load the prompt template if available, else use a minimal version
_PROMPT_PATH = Path(__file__).parent / "prompts" / "address_extraction.j2"
if _PROMPT_PATH.exists():
    from jinja2 import Template
    _PROMPT_TEMPLATE = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
    PROMPT = _PROMPT_TEMPLATE.render(ocr_text=SAMPLE_OCR)
else:
    PROMPT = f"Extract addresses from this WIPO patent OCR text as JSON:\n\n{SAMPLE_OCR}"


@dataclass
class BenchResult:
    concurrency: int
    num_ctx: int
    num_predict: int
    iterations: int
    wall_time_s: float
    avg_latency_s: float
    throughput_rps: float
    avg_tokens_in: float
    avg_tokens_out: float
    avg_eval_tps: float
    errors: int


def run_single(client: ollama.Client, model: str, num_ctx: int, num_predict: int) -> dict:
    """Run a single ollama.chat call and return timing info."""
    t0 = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        format=EXTRACTION_SCHEMA,
        options={
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        keep_alive="30m",
    )
    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "tokens_in": response.get("prompt_eval_count", 0),
        "tokens_out": response.get("eval_count", 0),
        "eval_duration_ns": response.get("eval_duration", 0),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration", 0),
    }


def benchmark(model: str, concurrency: int, num_ctx: int, num_predict: int, iterations: int) -> BenchResult:
    """Run `iterations` requests at `concurrency` parallelism."""
    client = ollama.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    results: list[dict] = []
    errors = 0
    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_single, client, model, num_ctx, num_predict)
            for _ in range(iterations)
        ]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
                print(f"  Error: {e}")

    wall_time = time.perf_counter() - t_wall_start

    if results:
        avg_latency = sum(r["elapsed"] for r in results) / len(results)
        eval_tps_values = [
            r["tokens_out"] / (r["eval_duration_ns"] / 1e9)
            for r in results if r["eval_duration_ns"] > 0
        ]
        avg_eval_tps = sum(eval_tps_values) / len(eval_tps_values) if eval_tps_values else 0
        avg_tok_in = sum(r["tokens_in"] for r in results) / len(results)
        avg_tok_out = sum(r["tokens_out"] for r in results) / len(results)
    else:
        avg_latency = avg_eval_tps = avg_tok_in = avg_tok_out = 0

    return BenchResult(
        concurrency=concurrency,
        num_ctx=num_ctx,
        num_predict=num_predict,
        iterations=iterations,
        wall_time_s=round(wall_time, 2),
        avg_latency_s=round(avg_latency, 3),
        throughput_rps=round(len(results) / wall_time, 2) if wall_time > 0 else 0,
        avg_tokens_in=round(avg_tok_in),
        avg_tokens_out=round(avg_tok_out),
        avg_eval_tps=round(avg_eval_tps, 1),
        errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama LLM throughput")
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency levels")
    parser.add_argument("--num-ctx", default="2048,4096,8192", help="Comma-separated num_ctx values")
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10, help="Requests per configuration")
    parser.add_argument("--warmup", action="store_true", help="Send 2 warmup requests first")
    parser.add_argument("--output", default=None, help="Save results as JSON to this path")
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    num_ctxs = [int(x) for x in args.num_ctx.split(",")]

    print(f"Model: {args.model}")
    print(f"Concurrency levels: {concurrencies}")
    print(f"Context sizes: {num_ctxs}")
    print(f"Iterations per config: {args.iterations}")
    print(f"Max output tokens: {args.num_predict}")
    print()

    if args.warmup:
        print("Warming up (loading model into GPU)...")
        client = ollama.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        run_single(client, args.model, 2048, 16)
        run_single(client, args.model, 2048, 16)
        print("Warmup done.\n")

    all_results: list[BenchResult] = []
    for num_ctx in num_ctxs:
        for conc in concurrencies:
            label = f"concurrency={conc}, num_ctx={num_ctx}"
            print(f"Testing: {label}, iterations={args.iterations}...")
            result = benchmark(args.model, conc, num_ctx, args.num_predict, args.iterations)
            all_results.append(result)
            print(
                f"  -> {result.throughput_rps} req/s, "
                f"avg_latency={result.avg_latency_s}s, "
                f"eval_tps={result.avg_eval_tps} tok/s, "
                f"errors={result.errors}"
            )

    # Print summary table
    print(f"\n{'=' * 95}")
    print(f"{'conc':>5} {'ctx':>6} {'iters':>6} {'wall_s':>7} {'lat_s':>7} {'rps':>7} "
          f"{'tok_in':>7} {'tok_out':>8} {'eval_tps':>9} {'err':>4}")
    print(f"{'-' * 95}")
    for r in all_results:
        print(
            f"{r.concurrency:>5} {r.num_ctx:>6} {r.iterations:>6} {r.wall_time_s:>7} "
            f"{r.avg_latency_s:>7} {r.throughput_rps:>7} {r.avg_tokens_in:>7.0f} "
            f"{r.avg_tokens_out:>8.0f} {r.avg_eval_tps:>9} {r.errors:>4}"
        )
    print(f"{'=' * 95}")

    # Find best configuration
    if all_results:
        best = max(all_results, key=lambda r: r.throughput_rps)
        print(f"\nBest config: concurrency={best.concurrency}, num_ctx={best.num_ctx} "
              f"-> {best.throughput_rps} req/s")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "num_predict": args.num_predict,
            "results": [
                {
                    "concurrency": r.concurrency,
                    "num_ctx": r.num_ctx,
                    "iterations": r.iterations,
                    "wall_time_s": r.wall_time_s,
                    "avg_latency_s": r.avg_latency_s,
                    "throughput_rps": r.throughput_rps,
                    "avg_tokens_in": r.avg_tokens_in,
                    "avg_tokens_out": r.avg_tokens_out,
                    "avg_eval_tps": r.avg_eval_tps,
                    "errors": r.errors,
                }
                for r in all_results
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
