"""OCR stage coordinator.

Implements the WIPO section-number heuristic:
  - Receives (row, images, pdf_metadata) from image_q
  - OCRs pages one by one using the loaded OCR model
  - After each page, runs the heuristic on accumulated text
  - Stops when the (72) inventor section is confirmed complete/absent
    or when max_pages is reached
  - Pushes (row, ocr_result, pdf_meta, ocr_meta) to text_q

OCR is blocking (GPU inference) and runs in a ThreadPoolExecutor.
This coordinator coroutine interacts with the event loop via await,
keeping the pipeline non-blocking for all other stages.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from models.ocr.base import OCRModel, OCRResult
from preprocessing.columns import detect_columns, split_top_left_column
from utils.logger import get_logger
from utils.status_tracker import StatusTracker
from utils.wipo import extract_sections, page_decision

logger = get_logger("ocr_worker");


def _run_ocr_sync(model: OCRModel, images: list) -> OCRResult:
    """Blocking wrapper — called from thread pool."""
    return model.run(images);


async def ocr_coordinator(
    image_q: asyncio.Queue,
    text_q: asyncio.Queue,
    ocr_model: OCRModel,
    n_upstream_sentinels: int,
    n_downstream_sentinels: int,
    config: dict,
    tracker: StatusTracker | None = None,
) -> None:
    """
    Single OCR coordinator coroutine.  Consumes from image_q, processes page
    by page with the WIPO heuristic, and feeds text_q.

    Terminates after receiving n_upstream_sentinels None values,
    then emits n_downstream_sentinels Nones to signal LLM workers.
    """
    n_workers = config["workers"]["ocr_workers"];
    loop = asyncio.get_event_loop();
    done_count = 0;

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ocr") as executor:
        while done_count < n_upstream_sentinels:
            item = await image_q.get();
            image_q.task_done();

            if item is None:
                done_count += 1;
                continue;

            row = item["row"];
            images: list | None = item["images"];
            pub_number = str(row.get("publication_number", "unknown"));

            if tracker:
                tracker.update("ocr_worker", status="running",
                               current_patent=pub_number,
                               queue_size=image_q.qsize());

            if item["error"] or not images:
                await text_q.put({
                    "row": row,
                    "ocr_text": "",
                    "pdf_meta": item,
                    "ocr_meta": {"error": item.get("error", "no images")},
                    "sections": set(),
                    "pages_used": 0,
                    "page_reason": "error",
                });
                continue;

            accumulated_text = "";
            accumulated_time = 0.0;
            pages_used = 0;
            page_reason = "not_reached";
            sections: set[int] = set();

            col_detection_cfg = config.get("column_detection", {});
            col_detection_enabled = col_detection_cfg.get("enabled", False);
            col_confidence_threshold = col_detection_cfg.get("confidence_threshold", 0.7);

            for page_idx, img in enumerate(images, start=1):
                t0 = time.perf_counter();
                try:
                    # Column splitting: only on page 1 (the bibliographic page)
                    ocr_img = img;
                    if page_idx == 1 and col_detection_enabled:
                        layout = detect_columns(img);
                        if layout.is_two_column and layout.confidence >= col_confidence_threshold:
                            ocr_img = split_top_left_column(img, layout);
                            logger.info(
                                f"[OCR] {pub_number}: column split applied "
                                f"(x={layout.split_x}, y={layout.separator_y}, "
                                f"conf={layout.confidence:.2f})"
                            );

                    result: OCRResult = await loop.run_in_executor(
                        executor, _run_ocr_sync, ocr_model, [ocr_img]
                    );
                    accumulated_text += ("\n" if accumulated_text else "") + result.text;
                    accumulated_time += result.elapsed_s;
                    pages_used = page_idx;
                    sections = extract_sections(accumulated_text);
                    decision, page_reason = page_decision(sections);

                    logger.debug(
                        f"[OCR] {pub_number} p{page_idx}: "
                        f"sections={sorted(sections)}, decision={decision}, reason={page_reason}"
                    );

                    if decision == "done":
                        break;

                except Exception as e:
                    import traceback
                    logger.warning(f"[OCR] {pub_number} p{page_idx}: {e}\n{traceback.format_exc()}");
                    page_reason = "ocr_error";
                    break;

            if tracker:
                ocr_stage = tracker.state["stages"]["ocr_worker"];
                tracker.update(
                    "ocr_worker",
                    completed=ocr_stage["completed"] + 1,
                    last_pages_used=pages_used,
                    last_page_reason=page_reason,
                    last_sections=[f"({s})" for s in sorted(sections)],
                    last_elapsed_s=round(accumulated_time, 3),
                    queue_size=text_q.qsize(),
                );
                tracker.record_timing("ocr", accumulated_time);

            section_strs = [f"({s})" for s in sorted(sections)];
            await text_q.put({
                "row": row,
                "ocr_text": accumulated_text,
                "pdf_meta": item,
                "ocr_meta": {
                    "model": ocr_model.model_name,
                    "elapsed_s": round(accumulated_time, 3),
                    "pages_processed": pages_used,
                    "char_count": len(accumulated_text),
                    "text_preview": accumulated_text[:200],
                },
                "sections": sections,
                "pages_used": pages_used,
                "page_reason": page_reason,
            });

    for _ in range(n_downstream_sentinels):
        await text_q.put(None);

    logger.info("OCR coordinator finished.");
