"""Output writer — final stage of the pipeline.

Appends one row to the CSV and one JSON line to the JSONL metadata file
per patent. Single-writer pattern: no locks needed.

Also manages temp file cleanup (batch mode) and tqdm progress updates.
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from models.llm.base import LLMResult
from utils.logger import get_logger
from utils.status_tracker import StatusTracker

logger = get_logger("output_writer");

_CSV_COLUMNS = [
    "publication_number", "title_text", "title_language",
    "application_number", "country_code", "publication_date",
    "filing_date", "grant_date", "priority_date",
    "inventor_names", "inventor_countries",
    "assignee_names", "assignee_countries",
    "ipc_codes", "cpc_codes",
    "google_patents_url",
    "inventors_with_address", "applicants_with_address",
    "addresses_found", "pdf_type", "pages_used", "page_reason", "sections_found",
    "ocr_model", "llm_provider", "ocr_elapsed_s", "llm_elapsed_s", "llm_cost_usd",
    "error",
];


def _build_url(pub_number: str) -> str:
    return f"https://patents.google.com/patent/{pub_number.replace('-', '')}";


def _make_csv_row(item: dict, config: dict) -> dict:
    row: dict = dict(item["row"]);
    pdf_meta: dict = item["pdf_meta"];
    ocr_meta: dict = item["ocr_meta"];
    llm: LLMResult = item["llm_result"];

    pub = str(row.get("publication_number", ""));
    row["google_patents_url"] = _build_url(pub);
    row["inventors_with_address"] = json.dumps(llm.inventors, ensure_ascii=False);
    row["applicants_with_address"] = json.dumps(llm.applicants, ensure_ascii=False);
    row["addresses_found"] = llm.found;
    row["pdf_type"] = pdf_meta.get("pdf_type", "");
    row["pages_used"] = item.get("pages_used", 0);
    row["page_reason"] = item.get("page_reason", "");
    row["sections_found"] = " ".join(item.get("sections_found", []));
    row["ocr_model"] = ocr_meta.get("model", "");
    row["llm_provider"] = f"{config['llm']['provider']}/{config['llm']['model']}";
    row["ocr_elapsed_s"] = ocr_meta.get("elapsed_s", "");
    row["llm_elapsed_s"] = round(llm.elapsed_s, 3);
    row["llm_cost_usd"] = llm.cost_usd if llm.cost_usd is not None else "";
    row["error"] = llm.error or "";

    return {col: row.get(col, "") for col in _CSV_COLUMNS};


def _make_meta_record(item: dict, config: dict, run_id: str) -> dict:
    pdf_meta = item["pdf_meta"];
    ocr_meta = item["ocr_meta"];
    llm: LLMResult = item["llm_result"];
    pub = str(item["row"].get("publication_number", ""));

    record: dict = {
        "patent_id": pub,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_url": pdf_meta.get("pdf_url"),
        "pdf_type": pdf_meta.get("pdf_type"),
        "pdf_elapsed_s": pdf_meta.get("elapsed_s"),
        "pages_extracted": len(pdf_meta.get("images") or []),
        "ocr": {
            "model": ocr_meta.get("model"),
            "device": config["ocr"].get("device"),
            "elapsed_s": ocr_meta.get("elapsed_s"),
            "pages_processed": item.get("pages_used"),
            "page_reason": item.get("page_reason"),
            "char_count": ocr_meta.get("char_count"),
            "text_preview": ocr_meta.get("text_preview"),
        },
        "llm": {
            "provider": config["llm"]["provider"],
            "model": config["llm"]["model"],
            "elapsed_s": round(llm.elapsed_s, 3),
            "tokens_in": llm.tokens_in,
            "tokens_out": llm.tokens_out,
            "cost_usd": llm.cost_usd,
            "retries": llm.retries,
        },
        "result": {
            "found": llm.found,
            "inventors_count": len(llm.inventors),
            "applicants_count": len(llm.applicants),
            "agents_count": len(llm.agents),
            "sections_detected": item.get("sections_found", []),
        },
        "error": llm.error,
    };

    record["llm_prompt"] = item.get("llm_prompt", "");

    if config["run_mode"] == "individual":
        record["thumbnail_paths"] = pdf_meta.get("thumbnail_paths", []);
        record["ocr_full_text"] = item.get("ocr_text", "");
        record["llm_raw_response"] = llm.raw_response;

    return record;


async def output_stage(
    result_q: asyncio.Queue,
    config: dict,
    run_id: str,
    total: int,
    n_upstream_sentinels: int,
    stats: dict,
    tracker: StatusTracker | None = None,
) -> None:
    """Single output-writer coroutine. Thread-safe because it's the only writer."""
    out_dir = Path(config["output"]["dir"]);
    out_dir.mkdir(parents=True, exist_ok=True);

    tmpl = config["output"]["filename_template"];
    batch = config.get("batch", {});
    yyyy = str(batch.get("year", "0000"));
    mm = str(batch.get("month", "00")).zfill(2);
    fname = tmpl.format(yyyy=yyyy, mm=mm);

    is_individual = config["run_mode"] == "individual";
    keep_files = is_individual and config.get("individual", {}).get("keep_temp_files", True);

    if is_individual:
        pat_id = config["individual"]["patent_id"].replace("-", "");
        meta_path = out_dir / "individual" / pat_id / "metadata.json";
        csv_path = out_dir / "individual" / pat_id / "result.csv";
        meta_path.parent.mkdir(parents=True, exist_ok=True);
    else:
        meta_path = out_dir / f"metadata_{fname}.jsonl";
        csv_path = out_dir / f"{fname}.csv";

    write_header = is_individual or not csv_path.exists();

    # Individual runs overwrite both files; batch runs append.
    file_mode = "w" if is_individual else "a";

    successes = 0;
    failures = 0;
    done_count = 0;

    pbar = tqdm(total=total, desc="Processing patents", unit="patent");

    with (
        open(csv_path, file_mode, newline="", encoding="utf-8") as csv_f,
        open(meta_path, file_mode, encoding="utf-8") as meta_f,
    ):
        writer = csv.DictWriter(csv_f, fieldnames=_CSV_COLUMNS);
        if write_header:
            writer.writeheader();

        while done_count < n_upstream_sentinels:
            item = await result_q.get();
            result_q.task_done();

            if item is None:
                done_count += 1;
                continue;

            csv_row = _make_csv_row(item, config);
            writer.writerow(csv_row);
            csv_f.flush();

            meta_rec = _make_meta_record(item, config, run_id);
            if is_individual:
                meta_f.write(json.dumps(meta_rec, indent=2, ensure_ascii=False));
            else:
                meta_f.write(json.dumps(meta_rec, ensure_ascii=False) + "\n");
            meta_f.flush();

            if item["llm_result"].found:
                successes += 1;
            else:
                failures += 1;

            if tracker:
                tracker.update(
                    "output_writer",
                    status="running",
                    completed=successes + failures,
                    successes=successes,
                    failures=failures,
                );

            pbar.update(1);
            pbar.set_postfix(ok=successes, fail=failures);

    pbar.close();

    stats.update({"successes": successes, "failures": failures, "total": successes + failures});
    logger.info(f"Output complete — {successes} OK, {failures} failed → {csv_path}");
