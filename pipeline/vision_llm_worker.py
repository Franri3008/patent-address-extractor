from __future__ import annotations

import asyncio
import time
from pathlib import Path

from jinja2 import Template

from models.llm.base import LLMResult
from models.llm.vision_base import VisionLLMModel
from utils.logger import get_logger

logger = get_logger("vision_llm_worker");

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "address_extraction_vision.j2";
_STOP_SECTION = "(72)";
_MAX_PAGES = 3;


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8");


def _merge_results(pages: list[LLMResult]) -> LLMResult:
    """Merge per-page LLMResults into a single result."""
    inventors: list[dict] = [];
    applicants: list[dict] = [];
    agents: list[dict] = [];
    sections: set[str] = set();
    total_tokens_in = 0;
    total_tokens_out = 0;
    total_elapsed = 0.0;
    found = False;

    for r in pages:
        inventors.extend(r.inventors);
        applicants.extend(r.applicants);
        agents.extend(r.agents);
        sections.update(r.sections_detected);
        total_tokens_in += r.tokens_in;
        total_tokens_out += r.tokens_out;
        total_elapsed += r.elapsed_s;
        if r.found:
            found = True;

    return LLMResult(
        inventors=inventors,
        applicants=applicants,
        agents=agents,
        sections_detected=sorted(sections),
        found=found,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        elapsed_s=round(total_elapsed, 3),
        cost_usd=None,
        retries=0,
    );


async def vision_llm_worker(
    image_q: asyncio.Queue,
    result_q: asyncio.Queue,
    vision_model: VisionLLMModel,
    config: dict,
    n_upstream_sentinels: int = 1,
    n_downstream_sentinels: int = 1,
) -> None:
    """
    Single vision LLM worker. Consumes from image_q (same format as ocr_worker),
    calls the vision LLM page-by-page, and pushes results directly to result_q.

    Waits for n_upstream_sentinels Nones from image_q before emitting
    n_downstream_sentinels Nones to result_q (mirrors ocr_coordinator pattern).
    """
    prompt_template = _load_prompt();
    max_pages: int = config.get("vision_llm", {}).get("max_pages", _MAX_PAGES);
    max_retries: int = config.get("vision_llm", {}).get("max_retries", 2);

    done_count = 0;

    while done_count < n_upstream_sentinels:
        item = await image_q.get();
        image_q.task_done();

        if item is None:
            done_count += 1;
            continue;

        row = item["row"];
        images = item["images"];
        pub_number = str(row.get("publication_number", "unknown"));

        if item["error"] or not images:
            llm_result = LLMResult(
                inventors=[], applicants=[], agents=[],
                sections_detected=[], found=False,
                tokens_in=0, tokens_out=0, elapsed_s=0.0,
                cost_usd=None, retries=0,
                error=item.get("error", "no images"),
            );
            await result_q.put(_build_output(row, llm_result, item, pages_used=0, page_reason="error"));
            continue;

        page_results: list[LLMResult] = [];
        pages_used = 0;
        page_reason = "max_pages";

        for page_idx, img in enumerate(images[:max_pages], start=1):
            page_result: LLMResult | None = None;
            for attempt in range(max_retries + 1):
                try:
                    page_result = await asyncio.to_thread(
                        vision_model.extract_addresses_from_image,
                        img, prompt_template, page_idx,
                    );
                    page_result.retries = attempt;
                    break;
                except Exception as e:
                    logger.warning(f"[VisionLLM] {pub_number} p{page_idx} attempt {attempt + 1}: {e}");
                    if attempt == max_retries:
                        page_result = LLMResult(
                            inventors=[], applicants=[], agents=[],
                            sections_detected=[], found=False,
                            tokens_in=0, tokens_out=0, elapsed_s=0.0,
                            cost_usd=None, retries=attempt,
                            error=str(e),
                        );

            page_results.append(page_result);
            pages_used = page_idx;

            logger.debug(
                f"[VisionLLM] {pub_number} p{page_idx}: "
                f"sections={page_result.sections_detected}, found={page_result.found}"
            );

            if _STOP_SECTION in page_result.sections_detected:
                page_reason = "done";
                break;

        llm_result = _merge_results(page_results);
        await result_q.put(_build_output(row, llm_result, item, pages_used, page_reason));

    for _ in range(n_downstream_sentinels):
        await result_q.put(None);


def _build_output(row: dict, llm_result: LLMResult, pdf_item: dict, pages_used: int, page_reason: str) -> dict:
    return {
        "row": row,
        "llm_result": llm_result,
        "pdf_meta": pdf_item,
        "ocr_meta": {"model": "vision_llm", "elapsed_s": 0.0, "pages_processed": 0, "char_count": 0},
        "pages_used": pages_used,
        "page_reason": page_reason,
        "sections_found": llm_result.sections_detected,
    };
