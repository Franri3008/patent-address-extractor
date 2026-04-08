#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

BASE_DIR        = Path(__file__).parent
TEST_DIR        = BASE_DIR / "test"
GROUND_TRUTH    = TEST_DIR / "ground_truth.csv"
CONFIG_DEFAULT  = BASE_DIR / "config.json"


def _normalize_name(name: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _extract_names(entities: list[dict]) -> set[str]:
    return {_normalize_name(e.get("name", "")) for e in entities if e.get("name")}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_outputs(expected: dict, actual: dict) -> dict:
    exp_found = bool(expected.get("found"))
    act_found = bool(actual.get("found"))
    found_match = exp_found == act_found

    exp_inv = expected.get("inventors") or []
    act_inv = actual.get("inventors") or []
    exp_app = expected.get("applicants") or []
    act_app = actual.get("applicants") or []

    inv_count_match = len(exp_inv) == len(act_inv)
    app_count_match = len(exp_app) == len(act_app)

    inv_score = _jaccard(_extract_names(exp_inv), _extract_names(act_inv))
    app_score = _jaccard(_extract_names(exp_app), _extract_names(act_app))

    overall = 0.2 * int(found_match) + 0.5 * inv_score + 0.3 * app_score

    if overall >= 0.9 and found_match:
        verdict = "pass"
    elif overall >= 0.5:
        verdict = "partial"
    else:
        verdict = "fail"

    return {
        "found_match":           found_match,
        "inventor_count_match":  inv_count_match,
        "applicant_count_match": app_count_match,
        "inventor_name_score":   round(inv_score, 3),
        "applicant_name_score":  round(app_score, 3),
        "overall_score":         round(overall, 3),
        "verdict":               verdict,
    }

async def _run_single(patent_id: str, config: dict) -> dict | None:
    """Run the extraction pipeline on one patent in individual mode."""
    import copy

    run_config = copy.deepcopy(config)
    run_config["run_mode"] = "individual"
    run_config.setdefault("individual", {})["patent_id"] = patent_id
    run_config.setdefault("individual", {})["save_thumbnails"] = False
    run_config.setdefault("individual", {})["keep_temp_files"] = False

    from pipeline.bq_fetcher import fetch_batch 
    from pipeline.llm_worker import llm_worker
    from pipeline.ocr_worker import ocr_coordinator
    from pipeline.output_writer import output_stage
    from pipeline.pdf_worker import pdf_worker
    from pipeline.vision_llm_worker import vision_llm_worker
    from models.llm import get_llm_model, get_vision_llm_model
    from models.ocr import get_ocr_model

    patent_rows = [{"publication_number": patent_id.replace("-", "")}]
    run_id = f"test_{patent_id}"

    workers_cfg = run_config["workers"]
    q_size  = workers_cfg["queue_max_size"]
    n_pdf   = workers_cfg["pdf_concurrency"]
    n_llm   = workers_cfg["llm_concurrency"]
    pipeline_mode = run_config.get("pipeline_mode", 0)

    patent_q: asyncio.Queue = asyncio.Queue(maxsize=q_size)
    image_q:  asyncio.Queue = asyncio.Queue(maxsize=q_size)
    result_q: asyncio.Queue = asyncio.Queue(maxsize=q_size)
    stats: dict[str, Any] = {}

    async def _feed():
        for row in patent_rows:
            await patent_q.put(row)
        for _ in range(n_pdf):
            await patent_q.put(None)

    pdf_tasks = [
        asyncio.create_task(_feed()),
        *[asyncio.create_task(pdf_worker(patent_q, image_q, run_config, n_sentinels_to_emit=1))
          for _ in range(n_pdf)],
    ]

    if pipeline_mode == 1:
        vision_model = get_vision_llm_model(run_config)
        middle_tasks = [asyncio.create_task(
            vision_llm_worker(image_q, result_q, vision_model, run_config,
                              n_upstream_sentinels=n_pdf, n_downstream_sentinels=n_llm)
        )]
        n_result_sentinels = n_llm
    else:
        text_q: asyncio.Queue = asyncio.Queue(maxsize=q_size)
        ocr_model = get_ocr_model(run_config)
        ocr_model.load()
        llm_model = get_llm_model(run_config)
        middle_tasks = [
            asyncio.create_task(
                ocr_coordinator(image_q, text_q, ocr_model, n_pdf, n_llm, run_config)
            ),
            *[asyncio.create_task(llm_worker(text_q, result_q, llm_model, run_config))
              for _ in range(n_llm)],
        ]
        n_result_sentinels = n_llm

    output_task = asyncio.create_task(
        output_stage(result_q, run_config, run_id, len(patent_rows),
                     n_result_sentinels, stats)
    )

    await asyncio.gather(*pdf_tasks, *middle_tasks, output_task)

    out_dir    = Path(run_config["output"]["dir"])
    meta_path  = out_dir / "individual" / patent_id.replace("-", "") / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        raw = meta.get("llm_raw_response", "")
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}
    return None


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH.exists():
        print(f"[test] Ground truth not found: {GROUND_TRUTH}")
        print("[test] Run `python main.py --review` to build it first.")
        return []

    rows = []
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_correct", "").lower() == "true":
                rows.append(row)
    return rows


def _fmt_score(score: float) -> str:
    pct = int(score * 100)
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    return f"{bar} {pct:3d}%"


def run_tests(config: dict | None = None) -> int:
    if config is None:
        config = _load_config(CONFIG_DEFAULT)

    rows = _load_ground_truth()
    if not rows:
        return 1

    print(f"\n{'═' * 70}")
    print(f"  Patent Pipeline — Regression Tests")
    print(f"  Ground truth: {len(rows)} patents (is_correct=true)")
    print(f"{'═' * 70}\n")

    results: list[dict] = []

    for i, row in enumerate(rows, 1):
        patent_id = row["patent_id"]
        print(f"[{i:>3}/{len(rows)}] {patent_id} ... ", end="", flush=True)

        try:
            expected_output = json.loads(row.get("llm_output_json", "{}") or "{}")
        except Exception:
            expected_output = {}

        try:
            actual_output = asyncio.run(_run_single(patent_id, config))
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "patent_id": patent_id,
                "verdict":   "error",
                "error":     str(exc),
                "overall_score": 0.0,
            })
            continue

        if actual_output is None:
            print("ERROR: pipeline produced no output")
            results.append({
                "patent_id": patent_id,
                "verdict":   "error",
                "error":     "no output",
                "overall_score": 0.0,
            })
            continue

        cmp = compare_outputs(expected_output, actual_output)
        verdict_sym = {"pass": "✓", "partial": "~", "fail": "✗", "error": "!"}.get(cmp["verdict"], "?")
        print(f"{verdict_sym}  score={_fmt_score(cmp['overall_score'])}  "
              f"found={'✓' if cmp['found_match'] else '✗'}  "
              f"inv={cmp['inventor_name_score']:.2f}  "
              f"app={cmp['applicant_name_score']:.2f}")

        results.append({"patent_id": patent_id, **cmp})

    total    = len(results)
    passed   = sum(1 for r in results if r["verdict"] == "pass")
    partial  = sum(1 for r in results if r["verdict"] == "partial")
    failed   = sum(1 for r in results if r["verdict"] in ("fail", "error"))
    avg_score = sum(r["overall_score"] for r in results) / total if total else 0.0

    print(f"\n{'─' * 70}")
    print(f"  Results:   {passed} pass  {partial} partial  {failed} fail  (total {total})")
    print(f"  Avg score: {_fmt_score(avg_score)}")

    if failed > 0:
        print(f"\n  Failed patents:")
        for r in results:
            if r["verdict"] in ("fail", "error"):
                msg = r.get("error") or f"score={r['overall_score']:.2f}";
                print(f"    • {r['patent_id']}  {msg}");

    print(f"{'═' * 70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    config_path = CONFIG_DEFAULT
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    try:
        cfg = _load_config(config_path)
    except FileNotFoundError:
        print(f"[test] Config not found: {config_path}")
        sys.exit(1)

    sys.exit(run_tests(cfg))
