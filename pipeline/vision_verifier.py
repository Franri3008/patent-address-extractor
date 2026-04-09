"""Vision LLM verification fallback for pipeline mode 0 (OCR + LLM).

When post-validation detects issues (missing known entities, section
mismatches, etc.), this module re-extracts addresses by sending the
original page image(s) to a vision-capable LLM (e.g. Gemma4 via Ollama).
The vision result replaces the OCR+LLM result if extraction succeeds.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from jinja2 import Template

from models.llm.base import LLMResult
from models.llm.vision_base import VisionLLMModel
from utils.logger import get_logger

logger = get_logger("vision_verifier");

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "address_verification.j2";


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8");


def _get_verification_model(config: dict) -> VisionLLMModel:
    """Instantiate a vision model using the verification config section.

    Maps config["verification"]["vision_llm"] to config["vision_llm"]
    so the existing OllamaVisionModel constructor works unchanged.
    """
    verify_cfg = config["verification"]["vision_llm"];
    # Build a synthetic config dict that OllamaVisionModel expects
    model_config = {**config, "vision_llm": verify_cfg};

    provider = verify_cfg.get("provider", "ollama");
    if provider == "ollama":
        from models.llm.ollama_vision import OllamaVisionModel
        return OllamaVisionModel(model_config);

    raise ValueError(f"Unknown verification vision LLM provider: '{provider}'");


async def verify_with_vision(
    images: list,
    first_pass_result: LLMResult,
    validation_warnings: list[str],
    row: dict,
    config: dict,
    template_vars: dict | None = None,
) -> LLMResult | None:
    """Re-extract addresses using a vision LLM when post-validation flags issues.

    Returns a new LLMResult on success, or None if verification is not
    triggered or fails.
    """
    if not validation_warnings:
        return None;

    verify_cfg = config.get("verification", {});
    if not verify_cfg.get("enabled", False):
        return None;

    if not images:
        return None;

    pub_number = str(row.get("publication_number", "unknown"));
    logger.info(
        f"[Verify] {pub_number}: triggering vision verification "
        f"({len(validation_warnings)} warning(s))"
    );

    try:
        vision_model = _get_verification_model(config);
    except Exception as e:
        logger.warning(f"[Verify] {pub_number}: failed to load vision model: {e}");
        return None;

    prompt_template = _load_prompt();

    # Build template context with first-pass result and known names
    import json
    first_pass_json = json.dumps({
        "inventors": first_pass_result.inventors,
        "applicants": first_pass_result.applicants,
        "agents": first_pass_result.agents,
    }, ensure_ascii=False, indent=2);

    vars_with_first_pass = {
        **(template_vars or {}),
        "first_pass_json": first_pass_json,
        "validation_warnings": validation_warnings,
    };

    # Send page 1 (the bibliographic page) to the vision model
    img = images[0];
    try:
        result: LLMResult = await asyncio.to_thread(
            vision_model.extract_addresses_from_image,
            img, prompt_template, 1, vars_with_first_pass,
        );
    except Exception as e:
        logger.warning(f"[Verify] {pub_number}: vision LLM call failed: {e}");
        return None;

    if result.error:
        logger.warning(f"[Verify] {pub_number}: vision LLM returned error: {result.error}");
        return None;

    logger.info(
        f"[Verify] {pub_number}: vision verification complete — "
        f"inventors={len(result.inventors)}, applicants={len(result.applicants)}"
    );
    return result;
