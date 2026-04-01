"""Anthropic API adapter (claude-3-5-sonnet, claude-3-haiku, etc.).

Set your API key in config["llm"]["api_key_env"], e.g. "ANTHROPIC_API_KEY".

Pricing: https://www.anthropic.com/pricing  (update _COST_PER_1K as needed)
"""
from __future__ import annotations

import os
import time

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from models.llm.ollama import _parse_response
from utils.logger import get_logger

logger = get_logger("llm.anthropic");

_COST_PER_1K: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-haiku-20240307":    (0.00025, 0.00125),
    "claude-3-opus-20240229":     (0.015,  0.075),
};


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    rates = _COST_PER_1K.get(model);
    if not rates:
        return None;
    in_rate, out_rate = rates;
    return round((tokens_in / 1000) * in_rate + (tokens_out / 1000) * out_rate, 6);


class AnthropicModel(LLMModel):
    def __init__(self, config: dict) -> None:
        self._model = config["llm"]["model"];
        self._temperature = config["llm"].get("temperature", 0.1);
        env_key = config["llm"].get("api_key_env") or "ANTHROPIC_API_KEY";
        self._api_key = os.environ.get(env_key);

    @property
    def provider_name(self) -> str:
        return "anthropic";

    @property
    def model_name(self) -> str:
        return self._model;

    def extract_addresses(self, ocr_text: str, prompt_template: str) -> LLMResult:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key);
        prompt = Template(prompt_template).render(ocr_text=ocr_text);

        t0 = time.perf_counter();
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        );
        elapsed = time.perf_counter() - t0;

        raw = response.content[0].text if response.content else "";
        tokens_in = response.usage.input_tokens;
        tokens_out = response.usage.output_tokens;
        cost = _estimate_cost(self._model, tokens_in, tokens_out);

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost);
