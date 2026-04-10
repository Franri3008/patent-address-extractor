#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from models.llm import get_llm_model, get_vision_llm_model
from models.ocr import get_ocr_model
from pipeline.bq_fetcher import fetch_batch
from pipeline.llm_worker import llm_worker
from pipeline.ocr_worker import ocr_coordinator
from pipeline.output_writer import output_stage
from pipeline.pdf_worker import pdf_worker
from pipeline.vision_llm_worker import vision_llm_worker
from utils.logger import get_logger
from utils.profiler import PipelineProfiler
from utils.reporter import print_individual_report, write_batch_report
from utils.status_tracker import StatusTracker

logger = get_logger("main");


def load_config(path: str) -> dict:
    p = Path(path);
    if not p.exists():
        logger.error(f"Config file not found: {path}. Copy config.example.json → config.json.");
        sys.exit(1);
    with open(p, encoding="utf-8") as f:
        return json.load(f);


async def run_pipeline(config: dict, patent_rows: list[dict], run_id: str,
                       tracker: StatusTracker | None = None,
                       profiler: PipelineProfiler | None = None) -> dict:
    """Assemble and run the async multi-stage pipeline."""
    workers_cfg = config["workers"];
    q_size: int = workers_cfg["queue_max_size"];
    n_pdf: int = workers_cfg["pdf_concurrency"];
    n_llm: int = workers_cfg["llm_concurrency"];
    pipeline_mode: int = config.get("pipeline_mode", 0);

    patent_q: asyncio.Queue = asyncio.Queue(maxsize=q_size);
    image_q: asyncio.Queue = asyncio.Queue(maxsize=q_size);
    result_q: asyncio.Queue = asyncio.Queue(maxsize=q_size);

    stats: dict = {};
    total = len(patent_rows);

    async def _feed() -> None:
        for row in patent_rows:
            await patent_q.put(row);
        for _ in range(n_pdf):
            await patent_q.put(None);

    async def _run_pdf() -> None:
        await asyncio.gather(
            _feed(),
            *[
                pdf_worker(patent_q, image_q, config, n_sentinels_to_emit=1,
                           tracker=tracker, profiler=profiler)
                for _ in range(n_pdf)
            ],
        );
        if tracker:
            tracker.update("pdf_worker", status="done", current_patent="");

    if pipeline_mode == 1:
        logger.info(f"pipeline_mode=1: loading vision LLM: {config['vision_llm']['provider']}/{config['vision_llm']['model']} ...");
        vision_model = get_vision_llm_model(config);

        async def _run_middle() -> None:
            await vision_llm_worker(
                image_q, result_q, vision_model, config,
                n_upstream_sentinels=n_pdf,
                n_downstream_sentinels=n_llm,
            );

        n_result_sentinels = n_llm;
    else:
        text_q: asyncio.Queue = asyncio.Queue(maxsize=q_size);

        logger.info(f"pipeline_mode=0: loading OCR model: {config['ocr']['model']} ...");
        ocr_model = get_ocr_model(config);
        ocr_model.load();

        logger.info(f"Loading LLM: {config['llm']['provider']}/{config['llm']['model']} ...");
        llm_model = get_llm_model(config);

        ocr_freed_event = asyncio.Event();

        async def _run_ocr() -> None:
            await ocr_coordinator(image_q, text_q, ocr_model, n_pdf, n_llm, config,
                                  tracker=tracker, profiler=profiler);
            keep_ocr = config.get("keep_ocr") or config.get("ocr", {}).get("keep_loaded", False);
            if not keep_ocr:
                ocr_model.unload();
                ocr_freed_event.set();
            if tracker:
                tracker.update("ocr_worker", status="done", current_patent="");

        async def _reload_llm_after_ocr() -> None:
            """Wait for OCR to free GPU, then reload the LLM to use freed VRAM."""
            await ocr_freed_event.wait();
            await asyncio.to_thread(llm_model.reload);

        async def _run_llm() -> None:
            await asyncio.gather(
                _reload_llm_after_ocr(),
                *[
                    llm_worker(text_q, result_q, llm_model, config,
                               tracker=tracker, profiler=profiler)
                    for _ in range(n_llm)
                ],
            );
            if tracker:
                tracker.update("llm_worker", status="done", current_patent="");

        async def _run_middle() -> None:
            await asyncio.gather(_run_ocr(), _run_llm());

        n_result_sentinels = n_llm;

    async def _run_output() -> None:
        await output_stage(result_q, config, run_id, total, n_result_sentinels, stats,
                           tracker=tracker, profiler=profiler);
        if tracker:
            tracker.update("output_writer", status="done");

    await asyncio.gather(
        asyncio.create_task(_run_pdf()),
        asyncio.create_task(_run_middle()),
        asyncio.create_task(_run_output()),
    );
    return stats;


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract inventor and applicant addresses from WO patents."
    );
    parser.add_argument("--config", default="config.json");
    parser.add_argument("--mode", choices=["individual", "batch"], default=None,
                        help="Override config run_mode.");
    parser.add_argument("--patent", metavar="ID", default=None,
                        help="Patent ID for individual mode (overrides config).");
    parser.add_argument("--pipeline-mode", type=int, choices=[0, 1], default=None,
                        help="Override config pipeline_mode (0=OCR+LLM, 1=vision LLM).");
    parser.add_argument("--review", action="store_true",
                        help="Start the review server to build the ground-truth dataset.");
    parser.add_argument("--viz", action="store_true",
                        help="Open the live pipeline dashboard while the run executes.");
    parser.add_argument("--test", action="store_true",
                        help="Run regression tests against test/ground_truth.csv.");
    parser.add_argument("--keep-ocr", action="store_true",
                        help="Keep OCR model/server loaded after OCR stage finishes "
                             "(default: unload to free GPU memory for LLM).");
    args = parser.parse_args();

    config = load_config(args.config);

    # ── Special modes: review server and test runner ──────────────────────────
    if args.review:
        from review_server import start_review_server
        start_review_server(config);
        return;

    if args.test:
        import test as test_module
        sys.exit(test_module.run_tests(config));

    if args.mode:
        config["run_mode"] = args.mode;
    if args.pipeline_mode is not None:
        config["pipeline_mode"] = args.pipeline_mode;
    if args.keep_ocr:
        config["keep_ocr"] = True;
    if args.patent:
        config["run_mode"] = "individual";
        config.setdefault("individual", {})["patent_id"] = args.patent;

    mode = config["run_mode"];
    out_dir = Path(config["output"]["dir"]);
    out_dir.mkdir(parents=True, exist_ok=True);

    tmpl = config["output"]["filename_template"];
    batch = config.get("batch", {});
    yyyy = str(batch.get("year", "0000"));
    mm = str(batch.get("month", "00")).zfill(2);
    fname = tmpl.format(yyyy=yyyy, mm=mm);
    run_id = f"{fname}_{mode}";

    dashboard_dir = Path("dashboard");
    dashboard_dir.mkdir(parents=True, exist_ok=True);

    _broadcast = None
    if args.viz:
        from viz_server import broadcast, start_viz_server
        start_viz_server()
        _broadcast = broadcast

    t_start = time.perf_counter();
    logger.info(f"Run started — mode={mode}, run_id={run_id}");

    if mode == "individual":
        patent_id = config["individual"]["patent_id"];
        logger.info(f"Individual mode — patent: {patent_id}");
        patent_rows = [{"publication_number": patent_id.replace("-", "")}];
    else:
        raw_csv = out_dir / f"raw_{fname}.csv";
        if raw_csv.exists():
            import pandas as pd
            logger.info(f"Loading cached BigQuery results from {raw_csv}");
            patent_rows = pd.read_csv(raw_csv).to_dict("records");
        else:
            patent_rows = fetch_batch(config, raw_csv);

        limit = config.get("batch", {}).get("limit");
        if limit:
            patent_rows = patent_rows[:int(limit)];
            logger.info(f"Limit applied: processing {len(patent_rows)} of available rows.");

    logger.info(f"Processing {len(patent_rows)} patents ...");

    profiler = PipelineProfiler();

    tracker = StatusTracker(
        dashboard_dir=dashboard_dir,
        run_id=run_id,
        run_mode=mode,
        pipeline_mode=config.get("pipeline_mode", 0),
        total_patents=len(patent_rows),
        broadcast_fn=_broadcast,
    );
    tracker.update("bq_fetch", status="done",
                   patents_fetched=len(patent_rows),
                   elapsed_s=round(time.perf_counter() - t_start, 3));

    stats = asyncio.run(run_pipeline(config, patent_rows, run_id, tracker=tracker, profiler=profiler));
    tracker.finish();

    elapsed = time.perf_counter() - t_start;
    logger.info(f"Pipeline complete in {elapsed:.1f}s.");

    # Print and save profiling report
    profiler.print_profile_report(elapsed);
    profile_path = out_dir / f"profile_{fname}.json";
    profiler.save_report(profile_path);

    if mode == "individual":
        patent_id = config["individual"]["patent_id"].replace("-", "");
        meta_path = out_dir / "individual" / patent_id / "metadata.json";
        print_individual_report(meta_path);
    else:
        meta_jsonl = out_dir / f"metadata_{fname}.jsonl";
        report_path = out_dir / f"report_{fname}.json";
        write_batch_report(meta_jsonl, report_path, stats, total_elapsed_s=elapsed);


if __name__ == "__main__":
    main();
