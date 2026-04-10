#!/usr/bin/env python3
"""Standalone LLM throughput benchmark — Ollama and vLLM backends.

Does NOT require real patents, OCR vLLM, or BigQuery — just a running
inference server with the target model loaded.

Tests concurrency levels to find optimal configuration for each backend.

Usage (Ollama):
    python benchmark_llm.py --backend ollama --model gemma4:e2b --warmup
    python benchmark_llm.py --backend ollama --concurrency 1,2,4,6 --num-ctx 2048,4096

Usage (vLLM):
    python benchmark_llm.py --backend vllm --model gemma --base-url http://localhost:8001/v1
    python benchmark_llm.py --backend vllm --concurrency 1,2,4,8,16

Head-to-head:
    python benchmark_llm.py --backend ollama --model gemma4:e2b --concurrency 1,4 --iterations 10
    python benchmark_llm.py --backend vllm --model gemma --base-url http://localhost:8001/v1 --concurrency 1,4,8,16 --iterations 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

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
    backend: str
    concurrency: int
    num_ctx: int
    max_tokens: int
    iterations: int
    wall_time_s: float
    avg_latency_s: float
    throughput_rps: float
    avg_tokens_in: float
    avg_tokens_out: float
    avg_eval_tps: float
    errors: int


# ── Ollama backend ──────────────────────────────────────────────────────────

def _ollama_single(client, model: str, num_ctx: int, num_predict: int) -> dict:
    t0 = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        format=EXTRACTION_SCHEMA,
        options={"temperature": 0.0, "num_ctx": num_ctx, "num_predict": num_predict},
        keep_alive="30m",
    )
    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "tokens_in": response.get("prompt_eval_count", 0),
        "tokens_out": response.get("eval_count", 0),
        "eval_duration_ns": response.get("eval_duration", 0),
    }


def bench_ollama(model: str, concurrency: int, num_ctx: int,
                 num_predict: int, iterations: int) -> BenchResult:
    import httpx
    import ollama
    client = ollama.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    results: list[dict] = []
    errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_ollama_single, client, model, num_ctx, num_predict)
            for _ in range(iterations)
        ]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
                print(f"  Error: {e}")
    wall = time.perf_counter() - t0
    return _summarise("ollama", concurrency, num_ctx, num_predict, iterations, results, errors, wall)


# ── vLLM backend ───────────────────────────────────────────────────────────

async def _vllm_single(client, model: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"guided_json": EXTRACTION_SCHEMA},
    )
    elapsed = time.perf_counter() - t0
    usage = response.usage
    return {
        "elapsed": elapsed,
        "tokens_in": usage.prompt_tokens if usage else 0,
        "tokens_out": usage.completion_tokens if usage else 0,
        "eval_duration_ns": 0,  # vLLM doesn't expose per-request eval duration
    }


async def _bench_vllm_async(model: str, base_url: str, concurrency: int,
                            max_tokens: int, iterations: int) -> tuple[list[dict], int, float]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=300)

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    errors = 0

    async def _task():
        nonlocal errors
        async with sem:
            try:
                r = await _vllm_single(client, model, max_tokens)
                results.append(r)
            except Exception as e:
                errors += 1
                print(f"  Error: {e}")

    t0 = time.perf_counter()
    await asyncio.gather(*[_task() for _ in range(iterations)])
    wall = time.perf_counter() - t0
    return results, errors, wall


def bench_vllm(model: str, base_url: str, concurrency: int,
               max_tokens: int, iterations: int) -> BenchResult:
    results, errors, wall = asyncio.run(
        _bench_vllm_async(model, base_url, concurrency, max_tokens, iterations)
    )
    return _summarise("vllm", concurrency, 0, max_tokens, iterations, results, errors, wall)


# ── Shared helpers ──────────────────────────────────────────────────────────

def _summarise(backend: str, concurrency: int, num_ctx: int, max_tokens: int,
               iterations: int, results: list[dict], errors: int, wall: float) -> BenchResult:
    if results:
        avg_lat = sum(r["elapsed"] for r in results) / len(results)
        eval_vals = [
            r["tokens_out"] / (r["eval_duration_ns"] / 1e9)
            for r in results if r["eval_duration_ns"] > 0
        ]
        avg_tps = sum(eval_vals) / len(eval_vals) if eval_vals else 0
        avg_in = sum(r["tokens_in"] for r in results) / len(results)
        avg_out = sum(r["tokens_out"] for r in results) / len(results)
    else:
        avg_lat = avg_tps = avg_in = avg_out = 0

    return BenchResult(
        backend=backend,
        concurrency=concurrency,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        iterations=iterations,
        wall_time_s=round(wall, 2),
        avg_latency_s=round(avg_lat, 3),
        throughput_rps=round(len(results) / wall, 2) if wall > 0 else 0,
        avg_tokens_in=round(avg_in),
        avg_tokens_out=round(avg_out),
        avg_eval_tps=round(avg_tps, 1),
        errors=errors,
    )


def print_table(all_results: list[BenchResult]) -> None:
    print(f"\n{'=' * 100}")
    print(f"{'backend':>8} {'conc':>5} {'ctx':>6} {'iters':>6} {'wall_s':>7} {'lat_s':>7} "
          f"{'rps':>7} {'tok_in':>7} {'tok_out':>8} {'eval_tps':>9} {'err':>4}")
    print(f"{'-' * 100}")
    for r in all_results:
        ctx_str = str(r.num_ctx) if r.num_ctx else "-"
        tps_str = f"{r.avg_eval_tps}" if r.avg_eval_tps else "-"
        print(
            f"{r.backend:>8} {r.concurrency:>5} {ctx_str:>6} {r.iterations:>6} "
            f"{r.wall_time_s:>7} {r.avg_latency_s:>7} {r.throughput_rps:>7} "
            f"{r.avg_tokens_in:>7.0f} {r.avg_tokens_out:>8.0f} {tps_str:>9} {r.errors:>4}"
        )
    print(f"{'=' * 100}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM throughput (Ollama / vLLM)")
    parser.add_argument("--backend", required=True, choices=["ollama", "vllm"])
    parser.add_argument("--model", default=None,
                        help="Model name (default: gemma4:e2b for ollama, gemma for vllm)")
    parser.add_argument("--base-url", default="http://localhost:8001/v1",
                        help="vLLM server URL (vllm backend only)")
    parser.add_argument("--concurrency", default="1,2,4",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--num-ctx", default="4096",
                        help="Comma-separated num_ctx values (ollama only)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    args = parser.parse_args()

    model = args.model
    if model is None:
        model = "gemma4:e2b" if args.backend == "ollama" else "gemma"

    concurrencies = [int(x) for x in args.concurrency.split(",")]
    num_ctxs = [int(x) for x in args.num_ctx.split(",")]

    print(f"Backend: {args.backend}")
    print(f"Model: {model}")
    print(f"Concurrency levels: {concurrencies}")
    if args.backend == "ollama":
        print(f"Context sizes: {num_ctxs}")
    print(f"Iterations per config: {args.iterations}")
    print(f"Max output tokens: {args.max_tokens}")
    print()

    if args.warmup:
        print("Warming up (loading model)...")
        if args.backend == "ollama":
            import httpx
            import ollama as _ollama
            c = _ollama.Client(timeout=httpx.Timeout(300.0, connect=10.0))
            _ollama_single(c, model, 2048, 16)
            _ollama_single(c, model, 2048, 16)
        else:
            asyncio.run(_bench_vllm_async(model, args.base_url, 1, 16, 2))
        print("Warmup done.\n")

    all_results: list[BenchResult] = []

    if args.backend == "ollama":
        for num_ctx in num_ctxs:
            for conc in concurrencies:
                print(f"Testing: concurrency={conc}, num_ctx={num_ctx}...")
                r = bench_ollama(model, conc, num_ctx, args.max_tokens, args.iterations)
                all_results.append(r)
                print(f"  -> {r.throughput_rps} req/s, lat={r.avg_latency_s}s, "
                      f"tps={r.avg_eval_tps}, err={r.errors}")
    else:
        for conc in concurrencies:
            print(f"Testing: concurrency={conc}...")
            r = bench_vllm(model, args.base_url, conc, args.max_tokens, args.iterations)
            all_results.append(r)
            print(f"  -> {r.throughput_rps} req/s, lat={r.avg_latency_s}s, err={r.errors}")

    print_table(all_results)

    if all_results:
        best = max(all_results, key=lambda r: r.throughput_rps)
        print(f"\nBest: {best.backend} concurrency={best.concurrency} -> {best.throughput_rps} req/s")

    if args.output:
        data = {"backend": args.backend, "model": model, "results": [
            {k: getattr(r, k) for k in r.__dataclass_fields__} for r in all_results
        ]}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
