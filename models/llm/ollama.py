"""Ollama local model adapter (e.g. gemma4:e2b).

Requires Ollama running locally:
  https://ollama.com/
  ollama pull gemma4:e2b

No API key needed. Cost is always None (local inference).

Performance tuning:
  - Set OLLAMA_NUM_PARALLEL=N on the server to allow N concurrent inferences.
  - num_ctx controls context window size (default 4096). Smaller = less VRAM per slot.
  - num_predict caps output tokens (default 1024). Prevents runaway generation.
  - keep_alive keeps model loaded between requests (default "30m").
"""
from __future__ import annotations

import json
import time

import httpx
import ollama
from jinja2 import Template

from models.llm.base import EXTRACTION_SCHEMA, LLMModel, LLMResult
from utils.logger import get_logger

logger = get_logger("llm.ollama")


class OllamaModel(LLMModel):
    def __init__(self, config: dict) -> None:
        self._model = config["llm"]["model"]
        self._temperature = config["llm"].get("temperature", 0.1)
        self._think = config["llm"].get("think", False)
        self._num_ctx = config["llm"].get("num_ctx", 4096)
        self._num_predict = config["llm"].get("num_predict", 1024)
        self._keep_alive = config["llm"].get("keep_alive", "30m")
        self._client = ollama.Client(
            host=config["llm"].get("base_url"),
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def reload(self) -> None:
        """Unload the model from GPU, then preload it so ollama re-evaluates available VRAM."""
        try:
            # keep_alive=0 tells ollama to unload the model immediately
            self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                keep_alive=0,
            )
            logger.info(f"Unloaded {self._model} from GPU.")
            # Preload: sending keep_alive with an empty prompt triggers a load
            self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_ctx": self._num_ctx},
                keep_alive=self._keep_alive,
            )
            logger.info(f"Reloaded {self._model} — ollama will use all available VRAM.")
        except Exception as e:
            logger.warning(f"Could not reload ollama model: {e}")

    def extract_addresses(self, ocr_text: str, prompt_template: str, template_vars: dict | None = None) -> LLMResult:
        prompt = Template(prompt_template).render(ocr_text=ocr_text, **(template_vars or {}))
        t0 = time.perf_counter()

        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=EXTRACTION_SCHEMA,
            options={
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
            think=self._think,
            keep_alive=self._keep_alive,
        )

        elapsed = time.perf_counter() - t0
        raw = response["message"]["content"]
        tokens_in = response.get("prompt_eval_count", 0)
        tokens_out = response.get("eval_count", 0)

        # Log truncation warning if prompt is close to context limit
        if tokens_in and tokens_in > 0.9 * self._num_ctx:
            logger.warning(
                f"Prompt tokens ({tokens_in}) near num_ctx limit ({self._num_ctx}). "
                f"Consider increasing num_ctx to avoid truncation."
            )

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost_usd=None)


def _parse_response(
    raw: str, elapsed: float, tokens_in: int, tokens_out: int, cost_usd: float | None
) -> LLMResult:
    """Parse LLM JSON response into a LLMResult."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
        addr_map = {a["id"]: a["address"] for a in data.get("addresses", [])}
        entities = data.get("entities", {})

        def resolve(entity_list: list) -> list:
            return [
                {"name": e["name"], "address": addr_map.get(e.get("address_id"))}
                for e in entity_list
            ]

        return LLMResult(
            inventors=resolve(entities.get("inventors", [])),
            applicants=resolve(entities.get("applicants", [])),
            agents=resolve(entities.get("agents", [])),
            sections_detected=[],
            found=bool(data.get("found", False)),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3),
            cost_usd=cost_usd,
            retries=0,
            raw_response=raw,
        )
    except json.JSONDecodeError as e:
        return LLMResult(
            inventors=[], applicants=[], agents=[],
            sections_detected=[], found=False,
            tokens_in=tokens_in, tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3), cost_usd=cost_usd, retries=0,
            error=f"JSON parse error: {e}",
            raw_response=raw,
        )
