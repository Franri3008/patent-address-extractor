"""Reporting utilities — individual and batch summaries."""
from __future__ import annotations

import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("reporter");


def print_individual_report(meta_path: Path) -> None:
    """Print a detailed console report for individual mode."""
    if not meta_path.exists():
        logger.warning(f"Metadata file not found: {meta_path}");
        return;

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f);

    pub = meta.get("patent_id", "?");
    ocr = meta.get("ocr", {});
    llm = meta.get("llm", {});
    result = meta.get("result", {});

    sep = "─" * 60;
    print(f"\n{sep}");
    print(f"  Individual Report — {pub}");
    print(sep);
    print(f"  PDF type          : {meta.get('pdf_type')}");
    print(f"  Pages extracted   : {meta.get('pages_extracted')}");
    print(f"  Pages used (OCR)  : {ocr.get('pages_processed')} ({ocr.get('page_reason', '')})");
    print(f"\n  OCR model         : {ocr.get('model')}  [{ocr.get('device')}]");
    print(f"  OCR time          : {ocr.get('elapsed_s'):.2f}s");
    print(f"  OCR chars         : {ocr.get('char_count')}");
    print(f"\n  LLM provider      : {llm.get('provider')} / {llm.get('model')}");
    print(f"  LLM time          : {llm.get('elapsed_s', 0):.2f}s");
    print(f"  Tokens in/out     : {llm.get('tokens_in')} / {llm.get('tokens_out')}");
    cost = llm.get("cost_usd");
    print(f"  Cost              : {'local (free)' if cost is None else f'${cost:.6f}'}");
    print(f"  Retries           : {llm.get('retries')}");
    print(f"\n  Addresses found   : {result.get('found')}");
    print(f"  Sections detected : {' '.join(result.get('sections_detected', []))}");
    print(f"  Inventors         : {result.get('inventors_count', 0)}");
    print(f"  Applicants        : {result.get('applicants_count', 0)}");
    print(f"  Agents            : {result.get('agents_count', 0)}");

    if meta.get("error"):
        print(f"\n  ⚠ Error           : {meta['error']}");

    thumbnails = meta.get("thumbnail_paths", []);
    if thumbnails:
        print(f"\n  Thumbnails        :");
        for p in thumbnails:
            print(f"    {p}");

    print(sep + "\n");


def write_batch_report(meta_jsonl_path: Path, report_path: Path, stats: dict) -> None:
    """Read the batch JSONL metadata and write a summary JSON report."""
    if not meta_jsonl_path.exists():
        return;

    records: list[dict] = [];
    with open(meta_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip();
            if line:
                try:
                    records.append(json.loads(line));
                except json.JSONDecodeError:
                    pass;

    if not records:
        return;

    total = len(records);
    found = sum(1 for r in records if r.get("result", {}).get("found"));
    failed = [r["patent_id"] for r in records if r.get("error")];
    pdf_types = {"vector": 0, "scanned": 0, "unknown": 0};
    pages_list: list[int] = [];
    ocr_times: list[float] = [];
    llm_times: list[float] = [];
    total_cost = 0.0;
    has_cost = False;

    for r in records:
        pt = r.get("pdf_type", "unknown");
        pdf_types[pt] = pdf_types.get(pt, 0) + 1;
        pages = r.get("ocr", {}).get("pages_processed");
        if pages:
            pages_list.append(pages);
        ot = r.get("ocr", {}).get("elapsed_s");
        if ot:
            ocr_times.append(ot);
        lt = r.get("llm", {}).get("elapsed_s");
        if lt:
            llm_times.append(lt);
        c = r.get("llm", {}).get("cost_usd");
        if c is not None:
            total_cost += c;
            has_cost = True;

    def _avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0;

    report = {
        "total_patents": total,
        "addresses_found": found,
        "addresses_not_found": total - found,
        "success_rate_pct": round(found / total * 100, 1) if total else 0,
        "failed_patents": failed,
        "pdf_type_breakdown": pdf_types,
        "avg_pages_per_patent": _avg(pages_list),
        "avg_ocr_time_s": _avg(ocr_times),
        "avg_llm_time_s": _avg(llm_times),
        "total_cost_usd": round(total_cost, 4) if has_cost else None,
    };

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False));
    logger.info(f"Batch report → {report_path}");

    sep = "─" * 60;
    print(f"\n{sep}");
    print(f"  Batch Report");
    print(sep);
    print(f"  Total patents     : {total}");
    print(f"  Addresses found   : {found} ({report['success_rate_pct']}%)");
    print(f"  Failed            : {len(failed)}");
    print(f"  PDF types         : {pdf_types}");
    print(f"  Avg pages used    : {report['avg_pages_per_patent']}");
    print(f"  Avg OCR time      : {report['avg_ocr_time_s']}s");
    print(f"  Avg LLM time      : {report['avg_llm_time_s']}s");
    cost_str = f"${report['total_cost_usd']:.4f}" if report["total_cost_usd"] is not None else "local (free)";
    print(f"  Total LLM cost    : {cost_str}");
    print(sep + "\n");
