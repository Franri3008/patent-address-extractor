"""Ollama local model adapter (e.g. gemma3:27b).

Requires Ollama running locally:
  https://ollama.com/
  ollama pull gemma3:27b

No API key needed. Cost is always None (local inference).
"""
from __future__ import annotations

import json
import time

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from utils.logger import get_logger

logger = get_logger("llm.ollama");


class OllamaModel(LLMModel):
    def __init__(self, config: dict) -> None:
        self._model = config["llm"]["model"];
        self._temperature = config["llm"].get("temperature", 0.1);

    @property
    def provider_name(self) -> str:
        return "ollama";

    @property
    def model_name(self) -> str:
        return self._model;

    def extract_addresses(self, ocr_text: str, prompt_template: str) -> LLMResult:
        import ollama

        prompt = Template(prompt_template).render(ocr_text=ocr_text);
        t0 = time.perf_counter();

        response = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self._temperature},
        );

        elapsed = time.perf_counter() - t0;
        raw = response["message"]["content"];
        tokens_in = response.get("prompt_eval_count", 0);
        tokens_out = response.get("eval_count", 0);

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost_usd=None);


def _parse_response(
    raw: str, elapsed: float, tokens_in: int, tokens_out: int, cost_usd: float | None
) -> LLMResult:
    """Parse LLM JSON response into a LLMResult."""
    cleaned = raw.strip();
    if cleaned.startswith("```"):
        lines = cleaned.split("\n");
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]);

    try:
        data = json.loads(cleaned);
        return LLMResult(
            inventors=data.get("inventors", []),
            applicants=data.get("applicants", []),
            agents=data.get("agents", []),
            sections_detected=data.get("sections_detected", []),
            found=bool(data.get("found", False)),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3),
            cost_usd=cost_usd,
            retries=0,
            raw_response=raw,
        );
    except json.JSONDecodeError as e:
        return LLMResult(
            inventors=[], applicants=[], agents=[],
            sections_detected=[], found=False,
            tokens_in=tokens_in, tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3), cost_usd=cost_usd, retries=0,
            error=f"JSON parse error: {e}",
            raw_response=raw,
        );
