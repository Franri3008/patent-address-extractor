"""Google Gemini API adapter (gemini-1.5-pro, gemini-1.5-flash, etc.).

Set your API key in config["llm"]["api_key_env"], e.g. "GOOGLE_API_KEY".

Pricing: https://ai.google.dev/pricing  (update _COST_PER_1K as needed)
"""
from __future__ import annotations

import os
import time

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from models.llm.ollama import _parse_response
from utils.logger import get_logger

logger = get_logger("llm.google");

_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gemini-1.5-pro":   (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash": (0.0001,   0.0004),
};


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    rates = _COST_PER_1K.get(model);
    if not rates:
        return None;
    in_rate, out_rate = rates;
    return round((tokens_in / 1000) * in_rate + (tokens_out / 1000) * out_rate, 6);


class GoogleModel(LLMModel):
    def __init__(self, config: dict) -> None:
        self._model_id = config["llm"]["model"];
        self._temperature = config["llm"].get("temperature", 0.1);
        api_key = config["llm"].get("api_key");
        if not api_key:
            env_key = config["llm"].get("api_key_env") or "GOOGLE_API_KEY";
            api_key = os.environ.get(env_key);
        import google.generativeai as genai
        genai.configure(api_key=api_key);
        self._genai = genai;

    @property
    def provider_name(self) -> str:
        return "google";

    @property
    def model_name(self) -> str:
        return self._model_id;

    def extract_addresses(self, ocr_text: str, prompt_template: str) -> LLMResult:
        model = self._genai.GenerativeModel(
            self._model_id,
            generation_config=self._genai.GenerationConfig(
                temperature=self._temperature,
                response_mime_type="application/json",
            ),
        );
        prompt = Template(prompt_template).render(ocr_text=ocr_text);

        t0 = time.perf_counter();
        response = model.generate_content(prompt);
        elapsed = time.perf_counter() - t0;

        raw = response.text or "";
        tokens_in = response.usage_metadata.prompt_token_count or 0;
        tokens_out = response.usage_metadata.candidates_token_count or 0;
        cost = _estimate_cost(self._model_id, tokens_in, tokens_out);

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost);
