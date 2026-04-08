"""LLM address-extraction worker (async).

Renders the Jinja2 prompt with OCR text and calls the configured LLM
to extract structured inventor/applicant/agent addresses.

Handles retries and propagates errors without crashing the pipeline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from utils.logger import get_logger
from utils.status_tracker import StatusTracker
from utils.wipo import extract_section_text, parse_known_names

logger = get_logger("llm_worker");

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "address_extraction.j2";


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8");


async def llm_worker(
    text_q: asyncio.Queue,
    result_q: asyncio.Queue,
    llm_model: LLMModel,
    config: dict,
    tracker: StatusTracker | None = None,
) -> None:
    """
    Single LLM worker coroutine. Consumes from text_q,
    calls LLM to extract addresses, pushes results to result_q.

    One sentinel None from text_q triggers one sentinel to result_q.
    """
    prompt_template = _load_prompt();
    max_retries: int = config["llm"].get("max_retries", 2);

    while True:
        item = await text_q.get();
        text_q.task_done();

        if item is None:
            await result_q.put(None);
            return;

        row = item["row"];
        ocr_text: str = item["ocr_text"];
        pub_number = str(row.get("publication_number", "unknown"));

        if tracker:
            tracker.update("llm_worker", status="running",
                           current_patent=pub_number,
                           queue_size=text_q.qsize());

        # Trim to target sections to minimise token count.
        # When (72) is missing from OCR (e.g. two-column layout garbled by
        # OCR), fall back to the full OCR text so the LLM can still attempt
        # extraction from whatever context is available.
        sec71 = extract_section_text(ocr_text, 71);
        sec72 = extract_section_text(ocr_text, 72);
        sec74 = extract_section_text(ocr_text, 74);

        if sec72:
            # Happy path: we have (72), trim aggressively.
            llm_input = "\n\n".join(filter(None, [sec71, sec72, sec74]));
        else:
            # (72) missing from OCR — send full text so the LLM has maximum
            # context (covers garbled column reads, truncated output, etc.).
            llm_input = ocr_text;

        template_vars = parse_known_names(row);

        logger.debug(
            f"[LLM] {pub_number}: OCR chars={len(ocr_text)}, trimmed chars={len(llm_input)}"
        );

        llm_result: LLMResult | None = None;
        ocr_error = item["ocr_meta"].get("error");

        if not ocr_text or ocr_error:
            llm_result = LLMResult(
                inventors=[], applicants=[], agents=[],
                sections_detected=[], found=False,
                tokens_in=0, tokens_out=0, elapsed_s=0.0,
                cost_usd=None, retries=0,
                error=ocr_error or "empty OCR text",
            );
        else:
            for attempt in range(max_retries + 1):
                try:
                    llm_result = await asyncio.to_thread(
                        llm_model.extract_addresses, llm_input, prompt_template, template_vars
                    );
                    llm_result.retries = attempt;
                    break;
                except Exception as e:
                    logger.warning(f"[LLM] {pub_number} attempt {attempt + 1}: {e}");
                    if attempt == max_retries:
                        llm_result = LLMResult(
                            inventors=[], applicants=[], agents=[],
                            sections_detected=[], found=False,
                            tokens_in=0, tokens_out=0, elapsed_s=0.0,
                            cost_usd=None, retries=attempt,
                            error=str(e),
                        );

        rendered_prompt = Template(prompt_template).render(ocr_text=llm_input, **template_vars);

        if tracker and llm_result:
            llm_stage = tracker.state["stages"]["llm_worker"];
            tracker.update(
                "llm_worker",
                completed=llm_stage["completed"] + 1,
                errors=llm_stage["errors"] + (1 if llm_result.error else 0),
                last_elapsed_s=round(llm_result.elapsed_s, 3),
                last_raw_response=llm_result.raw_response or "",
                last_result_preview={
                    "found": llm_result.found,
                    "inventors_count": len(llm_result.inventors),
                    "applicants_count": len(llm_result.applicants),
                    "agents_count": len(llm_result.agents),
                },
                queue_size=result_q.qsize(),
            );
            tracker.record_timing("llm", llm_result.elapsed_s);

            # Save used page images for comparison view
            images = item["pdf_meta"].get("images");
            pages_used = item.get("pages_used", 0);
            if images and pages_used:
                page_paths = tracker.save_page_images(pub_number, images, pages_used);
                tracker.set_comparison(pub_number, page_paths, {
                    "inventors": llm_result.inventors,
                    "applicants": llm_result.applicants,
                    "agents": llm_result.agents,
                });

        await result_q.put({
            "row": row,
            "llm_result": llm_result,
            "pdf_meta": item["pdf_meta"],
            "ocr_meta": item["ocr_meta"],
            "pages_used": item["pages_used"],
            "page_reason": item["page_reason"],
            "sections_found": [f"({s})" for s in sorted(item.get("sections", []))],
            "llm_prompt": rendered_prompt,
        });
