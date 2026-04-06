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
from utils.wipo import extract_section_text

logger = get_logger("llm_worker");

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "address_extraction.j2";


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8");


async def llm_worker(
    text_q: asyncio.Queue,
    result_q: asyncio.Queue,
    llm_model: LLMModel,
    config: dict,
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

        # Trim to only sections (71) and (72) to minimise token count.
        # Falls back to full OCR text if neither section is detected.
        sections_text = "\n\n".join(filter(None, [
            extract_section_text(ocr_text, 71),
            extract_section_text(ocr_text, 72),
        ]));
        llm_input = sections_text or ocr_text;

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
                        llm_model.extract_addresses, llm_input, prompt_template
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

        await result_q.put({
            "row": row,
            "llm_result": llm_result,
            "pdf_meta": item["pdf_meta"],
            "ocr_meta": item["ocr_meta"],
            "pages_used": item["pages_used"],
            "page_reason": item["page_reason"],
            "sections_found": [f"({s})" for s in sorted(item.get("sections", []))],
        });
