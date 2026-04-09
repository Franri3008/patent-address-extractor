"""OCR stage coordinator — concurrent patent processing.

Implements the WIPO section-number heuristic with cross-patent parallelism:
  - Receives (row, images, pdf_metadata) from image_q
  - Spawns an async task per patent, limited by a semaphore
  - Each task OCRs pages one by one, running the WIPO heuristic per page
  - Stops when the (72) inventor section is confirmed complete/absent
    or when max_pages is reached
  - Pushes (row, ocr_result, pdf_meta, ocr_meta) to text_q

Cross-patent parallelism means vLLM can batch concurrent OCR requests
from different patents, while the per-patent page heuristic is preserved
(page N+1 depends on page N within each patent).

The semaphore limits concurrent vLLM requests (not patents). Between
pages (while running page_decision), the semaphore is released so
another patent's request can proceed.
"""
from __future__ import annotations

import asyncio
import time

from models.ocr.base import OCRModel, OCRResult
from preprocessing.columns import detect_columns, split_top_left_column
from utils.logger import get_logger
from utils.ocr_cache import OCRCache
from utils.profiler import PipelineProfiler
from utils.status_tracker import StatusTracker
from utils.wipo import extract_sections, page_decision

logger = get_logger("ocr_worker")


async def _process_single_patent(
    item: dict,
    ocr_model: OCRModel,
    semaphore: asyncio.Semaphore,
    config: dict,
    tracker: StatusTracker | None,
    profiler: PipelineProfiler | None,
    ocr_cache: OCRCache | None,
) -> dict:
    """Process one patent's OCR with the WIPO page heuristic.

    Acquires the semaphore only during the actual vLLM request,
    releasing it between pages so other patents can proceed.
    """
    row = item["row"]
    images: list | None = item["images"]
    pub_number = str(row.get("publication_number", "unknown"))

    if tracker:
        tracker.update("ocr_worker", status="running",
                       current_patent=pub_number,
                       queue_size=0)

    if item["error"] or not images:
        if profiler:
            profiler.record_ocr_done(pub_number, 0.0, 0)
        return {
            "row": row,
            "ocr_text": "",
            "pdf_meta": item,
            "ocr_meta": {"error": item.get("error", "no images")},
            "sections": set(),
            "pages_used": 0,
            "page_reason": "error",
        }

    accumulated_text = ""
    accumulated_time = 0.0
    pages_used = 0
    page_reason = "not_reached"
    sections: set[int] = set()

    col_detection_cfg = config.get("column_detection", {})
    col_detection_enabled = col_detection_cfg.get("enabled", False)
    col_confidence_threshold = col_detection_cfg.get("confidence_threshold", 0.7)

    for page_idx, img in enumerate(images, start=1):
        try:
            # Column splitting: only on page 1 (the bibliographic page)
            ocr_img = img
            if page_idx == 1 and col_detection_enabled:
                layout = detect_columns(img)
                if layout.is_two_column and layout.confidence >= col_confidence_threshold:
                    ocr_img = split_top_left_column(img, layout)
                    logger.info(
                        f"[OCR] {pub_number}: column split applied "
                        f"(x={layout.split_x}, y={layout.separator_y}, "
                        f"conf={layout.confidence:.2f})"
                    )

            # Check cache first
            cached_text = ocr_cache.get(pub_number, page_idx) if ocr_cache else None
            if cached_text is not None:
                page_text = cached_text
                page_time = 0.0
                logger.debug(f"[OCR] {pub_number} p{page_idx}: cache hit")
            else:
                # Acquire semaphore only during the vLLM request
                t_page0 = time.perf_counter()
                async with semaphore:
                    result: OCRResult = await ocr_model.run_async([ocr_img])
                page_time = time.perf_counter() - t_page0
                page_text = result.text
                # Write to cache
                if ocr_cache:
                    ocr_cache.put(pub_number, page_idx, page_text)

            accumulated_text += ("\n" if accumulated_text else "") + page_text
            accumulated_time += page_time
            pages_used = page_idx

            if profiler:
                profiler.record_ocr_page(pub_number, page_time)

            sections = extract_sections(accumulated_text)
            decision, page_reason = page_decision(sections)

            logger.debug(
                f"[OCR] {pub_number} p{page_idx}: "
                f"sections={sorted(sections)}, decision={decision}, reason={page_reason}"
            )

            if decision == "done":
                break

        except Exception as e:
            import traceback
            logger.warning(f"[OCR] {pub_number} p{page_idx}: {e}\n{traceback.format_exc()}")
            page_reason = "ocr_error"
            break

    if profiler:
        profiler.record_ocr_done(pub_number, accumulated_time, pages_used)

    if tracker:
        ocr_stage = tracker.state["stages"]["ocr_worker"]
        tracker.update(
            "ocr_worker",
            completed=ocr_stage["completed"] + 1,
            last_pages_used=pages_used,
            last_page_reason=page_reason,
            last_sections=[f"({s})" for s in sorted(sections)],
            last_elapsed_s=round(accumulated_time, 3),
            queue_size=0,
        )
        tracker.record_timing("ocr", accumulated_time)

    section_strs = [f"({s})" for s in sorted(sections)]
    return {
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
    }


async def ocr_coordinator(
    image_q: asyncio.Queue,
    text_q: asyncio.Queue,
    ocr_model: OCRModel,
    n_upstream_sentinels: int,
    n_downstream_sentinels: int,
    config: dict,
    tracker: StatusTracker | None = None,
    profiler: PipelineProfiler | None = None,
) -> None:
    """
    Concurrent OCR coordinator. Dispatches patent tasks behind a semaphore
    to enable cross-patent parallelism while preserving per-patent WIPO heuristic.

    Terminates after receiving n_upstream_sentinels None values,
    then emits n_downstream_sentinels Nones to signal LLM workers.
    """
    ocr_concurrency = config["workers"].get("ocr_concurrency", 4)
    semaphore = asyncio.Semaphore(ocr_concurrency)

    # Initialize OCR cache
    ocr_cfg = config.get("ocr", {})
    cache_enabled = ocr_cfg.get("cache_enabled", False)
    cache_dir = ocr_cfg.get("cache_dir", "output/ocr_cache")
    ocr_cache = OCRCache(cache_dir, enabled=cache_enabled)

    done_count = 0
    pending: list[asyncio.Task] = []

    while done_count < n_upstream_sentinels:
        item = await image_q.get()
        image_q.task_done()

        if item is None:
            done_count += 1
            continue

        # Record queue wait time
        if profiler:
            enqueue_t = item.get("_enqueue_t")
            pub_number = str(item["row"].get("publication_number", "unknown"))
            if enqueue_t:
                profiler.record_ocr_start(pub_number, queue_wait_s=time.perf_counter() - enqueue_t)
            profiler.ocr_inflight_inc()

        # Spawn a task for this patent
        task = asyncio.create_task(
            _process_single_patent(item, ocr_model, semaphore, config, tracker, profiler, ocr_cache)
        )
        pending.append(task)

        # Drain completed tasks to push results downstream without unbounded growth
        still_pending = []
        for t in pending:
            if t.done():
                try:
                    result = t.result()
                    result["_enqueue_t"] = time.perf_counter()
                    await text_q.put(result)
                except Exception as e:
                    logger.warning(f"[OCR] task error: {e}")
                if profiler:
                    profiler.ocr_inflight_dec()
            else:
                still_pending.append(t)
        pending = still_pending

    # Wait for all remaining in-flight patent tasks
    for t in asyncio.as_completed(pending):
        try:
            result = await t
            result["_enqueue_t"] = time.perf_counter()
            await text_q.put(result)
        except Exception as e:
            logger.warning(f"[OCR] task error: {e}")
        if profiler:
            profiler.ocr_inflight_dec()

    if cache_enabled:
        logger.info(f"OCR cache stats: {ocr_cache.stats}")

    for _ in range(n_downstream_sentinels):
        await text_q.put(None)

    logger.info("OCR coordinator finished.")
