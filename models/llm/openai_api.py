"""OpenAI API adapter (gpt-4o, gpt-4o-mini, etc.).

Set your API key in config["llm"]["api_key_env"], e.g. "OPENAI_API_KEY",
or export it as an environment variable before running.

Pricing: https://openai.com/pricing  (update _COST_PER_1K as needed)
"""
from __future__ import annotations

import os
import time

from jinja2 import Template

from models.llm.base import LLMModel, LLMResult
from models.llm.ollama import _parse_response
from utils.logger import get_logger

logger = get_logger("llm.openai");

_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o":       (0.0025, 0.010),
    "gpt-4o-mini":  (0.00015, 0.0006),
    "gpt-4-turbo":  (0.010,  0.030),
};


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    rates = _COST_PER_1K.get(model);
    if not rates:
        return None;
    in_rate, out_rate = rates;
    return round((tokens_in / 1000) * in_rate + (tokens_out / 1000) * out_rate, 6);


class OpenAIModel(LLMModel):
    def __init__(self, config: dict) -> None:
        self._model = config["llm"]["model"];
        self._temperature = config["llm"].get("temperature", 0.1);
        self._api_key = config["llm"].get("api_key");
        if not self._api_key:
            env_key = config["llm"].get("api_key_env") or "OPENAI_API_KEY";
            self._api_key = os.environ.get(env_key);
        if not self._api_key:
            raise ValueError(
                f"Missing OpenAI API key. Set the '{env_key}' environment variable."
            );
        self._base_url = config["llm"].get("base_url");

    @property
    def provider_name(self) -> str:
        return "openai";

    @property
    def model_name(self) -> str:
        return self._model;

    def extract_addresses(self, ocr_text: str, prompt_template: str) -> LLMResult:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key, base_url=self._base_url);
        prompt = Template(prompt_template).render(ocr_text=ocr_text);

        t0 = time.perf_counter();
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        );
        elapsed = time.perf_counter() - t0;

        raw = response.choices[0].message.content or "";
        usage = response.usage;
        tokens_in = usage.prompt_tokens if usage else 0;
        tokens_out = usage.completion_tokens if usage else 0;
        cost = _estimate_cost(self._model, tokens_in, tokens_out);

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost);
