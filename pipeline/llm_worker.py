"""LLM address-extraction worker (async).

Renders the Jinja2 prompt with OCR text and calls the configured LLM
to extract structured inventor/applicant/agent addresses.

Handles retries and propagates errors without crashing the pipeline.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from pipeline.vision_verifier import verify_with_vision
from utils.logger import get_logger
from utils.profiler import PipelineProfiler
from utils.status_tracker import StatusTracker
from utils.validators import run_all_validations
from utils.wipo import extract_section_text, parse_known_names

logger = get_logger("llm_worker");

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "address_extraction.j2";

# Conservative chars-per-token estimate. Real ratio is ~3.5-4 for English,
# lower for CJK/mixed scripts. We pick 3 to stay safely under the limit.
_CHARS_PER_TOKEN = 3;

# Safety margin (in tokens) between our estimate and the model's hard limit,
# to absorb estimation error from the char-based approximation.
_TOKEN_SAFETY_MARGIN = 128;


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8");


def _truncate_for_context(
    llm_input: str,
    prompt_template: str,
    template_vars: dict,
    max_model_len: int,
    max_output_tokens: int,
    pub_number: str,
) -> str:
    """Truncate ``llm_input`` so the rendered prompt fits the model context.

    Uses a char-based token estimate — good enough given the safety margin.
    Returns ``llm_input`` unchanged if it already fits.
    """
    if not llm_input:
        return llm_input;

    # Token budget available for the input prompt.
    input_token_budget = max_model_len - max_output_tokens - _TOKEN_SAFETY_MARGIN;
    if input_token_budget <= 0:
        logger.warning(
            f"[LLM] {pub_number}: max_model_len ({max_model_len}) too small for "
            f"max_tokens ({max_output_tokens}); skipping truncation"
        );
        return llm_input;

    # Rendered-prompt overhead (template + known names, without OCR text).
    overhead_chars = len(
        Template(prompt_template).render(ocr_text="", **template_vars)
    );

    # Total char budget for the full rendered prompt.
    total_char_budget = input_token_budget * _CHARS_PER_TOKEN;
    ocr_char_budget = total_char_budget - overhead_chars;

    if ocr_char_budget <= 0:
        logger.warning(
            f"[LLM] {pub_number}: prompt overhead ({overhead_chars} chars) already "
            f"exceeds context budget; sending minimal input"
        );
        return "";

    if len(llm_input) <= ocr_char_budget:
        return llm_input;

    logger.info(
        f"[LLM] {pub_number}: truncating OCR text {len(llm_input)} -> {ocr_char_budget} "
        f"chars to fit {max_model_len}-token context window"
    );
    return llm_input[:ocr_char_budget];


async def llm_worker(
    text_q: asyncio.Queue,
    result_q: asyncio.Queue,
    llm_model: LLMModel,
    config: dict,
    tracker: StatusTracker | None = None,
    profiler: PipelineProfiler | None = None,
) -> None:
    """
    Single LLM worker coroutine. Consumes from text_q,
    calls LLM to extract addresses, pushes results to result_q.

    One sentinel None from text_q triggers one sentinel to result_q.
    """
    prompt_template = _load_prompt();
    max_retries: int = config["llm"].get("max_retries", 2);
    max_model_len: int | None = config["llm"].get("max_model_len");
    max_output_tokens: int = config["llm"].get("max_tokens", 1024);

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

        # Record LLM queue wait time
        if profiler:
            enqueue_t = item.get("_enqueue_t")
            if enqueue_t:
                profiler.record_llm_start(pub_number, queue_wait_s=time.perf_counter() - enqueue_t);
            profiler.llm_inflight_inc();

        # Trim to target sections to minimise token count.
        # When (72) is missing from OCR (e.g. two-column layout garbled by
        # OCR), fall back to the full OCR text so the LLM can still attempt
        # extraction from whatever context is available.
        sec71 = extract_section_text(ocr_text, 71);
        sec72 = extract_section_text(ocr_text, 72);

        if sec72:
            # Happy path: we have (72), trim aggressively.
            llm_input = "\n\n".join(filter(None, [sec71, sec72]));
        else:
            # (72) missing from OCR — send full text so the LLM has maximum
            # context (covers garbled column reads, truncated output, etc.).
            llm_input = ocr_text;

        template_vars = parse_known_names(row);

        # Keep the rendered prompt within the model's context window.
        if max_model_len:
            llm_input = _truncate_for_context(
                llm_input, prompt_template, template_vars,
                max_model_len, max_output_tokens, pub_number,
            );

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
            # Use native async if the model supports it (vLLM), else fall back to thread
            _has_async = hasattr(llm_model, "extract_addresses_async")
            for attempt in range(max_retries + 1):
                try:
                    if _has_async:
                        llm_result = await llm_model.extract_addresses_async(
                            llm_input, prompt_template, template_vars
                        );
                    else:
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

        # Record LLM profiling
        if profiler and llm_result:
            profiler.record_llm_done(
                pub_number, llm_result.elapsed_s,
                llm_result.tokens_in, llm_result.tokens_out,
            );
            profiler.llm_inflight_dec();

        rendered_prompt = Template(prompt_template).render(ocr_text=llm_input, **template_vars);

        # --- Post-validation (deterministic, no extra LLM calls) ---
        validation_warnings: list[str] = [];
        vision_verified = False;
        if (
            config["llm"].get("post_validation", {}).get("enabled", False)
            and llm_result
            and not llm_result.error
        ):
            validation_warnings = run_all_validations(
                llm_result,
                known_applicants=template_vars.get("known_applicants"),
                known_inventors=template_vars.get("known_inventors"),
                ocr_sections=item.get("sections"),
            );
            if validation_warnings:
                logger.info(
                    f"[LLM] {pub_number}: validation warnings: {validation_warnings}"
                );

                # --- Vision LLM verification fallback ---
                images = item["pdf_meta"].get("images");
                if images and config.get("verification", {}).get("enabled", False):
                    verified = await verify_with_vision(
                        images, llm_result, validation_warnings,
                        row, config, template_vars,
                    );
                    if verified:
                        llm_result = verified;
                        vision_verified = True;

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
            "llm_prompt": rendered_prompt,
            "validation_warnings": validation_warnings,
            "vision_verified": vision_verified,
        });
