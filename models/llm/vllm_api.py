"""vLLM adapter for text-only LLM inference (e.g. Gemma served via vLLM).

Uses the OpenAI-compatible API exposed by vLLM, with guided_json
for structured output (vLLM's native constrained decoding — faster
than prompt-only JSON and more reliable than Ollama's format param).

Start the server on a SEPARATE port from the OCR vLLM instance:

    vllm serve google/gemma-3-4b-it \
        --port 8001 \
        --served-model-name gemma \
        --gpu-memory-utilization 0.45 \
        --max-num-seqs 16 \
        --max-model-len 4096 \
        --dtype auto

Config keys (under "llm"):
    provider   : "vllm"
    model      : "gemma"                        (must match --served-model-name)
    base_url   : "http://localhost:8001/v1"      (must include /v1)
    temperature: 0.0
    max_tokens : 1024                            (caps output length)
"""
from __future__ import annotations

import json
import time

from jinja2 import Template

from models.llm.base import EXTRACTION_SCHEMA, LLMModel, LLMResult
from utils.logger import get_logger

logger = get_logger("llm.vllm")


class VLLMModel(LLMModel):
    """LLM adapter that talks to a vLLM server via its OpenAI-compatible API."""

    def __init__(self, config: dict) -> None:
        llm_cfg = config["llm"]
        self._model = llm_cfg["model"]
        self._temperature = llm_cfg.get("temperature", 0.0)
        self._max_tokens = llm_cfg.get("max_tokens", 1024)
        base_url = llm_cfg.get("base_url")
        if not base_url:
            raise ValueError(
                "vLLM provider requires 'base_url' in config['llm'], "
                "e.g. 'http://localhost:8001/v1'"
            )
        self._base_url = base_url
        self._client = None
        self._async_client = None
        self._load_clients()

    def _load_clients(self) -> None:
        from openai import AsyncOpenAI, OpenAI

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=self._base_url,
            timeout=300,
        )
        self._async_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=self._base_url,
            timeout=300,
        )
        logger.info(
            f"vLLM LLM client ready — server={self._base_url}, model={self._model}"
        )

    @property
    def provider_name(self) -> str:
        return "vllm"

    @property
    def model_name(self) -> str:
        return self._model

    def extract_addresses(
        self, ocr_text: str, prompt_template: str, template_vars: dict | None = None
    ) -> LLMResult:
        prompt = Template(prompt_template).render(
            ocr_text=ocr_text, **(template_vars or {})
        )
        t0 = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"guided_json": EXTRACTION_SCHEMA},
        )

        elapsed = time.perf_counter() - t0
        raw = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost_usd=None)

    async def extract_addresses_async(
        self, ocr_text: str, prompt_template: str, template_vars: dict | None = None
    ) -> LLMResult:
        """Native async version — avoids asyncio.to_thread overhead."""
        prompt = Template(prompt_template).render(
            ocr_text=ocr_text, **(template_vars or {})
        )
        t0 = time.perf_counter()

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"guided_json": EXTRACTION_SCHEMA},
        )

        elapsed = time.perf_counter() - t0
        raw = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return _parse_response(raw, elapsed, tokens_in, tokens_out, cost_usd=None)


def _parse_response(
    raw: str, elapsed: float, tokens_in: int, tokens_out: int, cost_usd: float | None
) -> LLMResult:
    """Parse JSON response from vLLM into LLMResult."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

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
            inventors=[],
            applicants=[],
            agents=[],
            sections_detected=[],
            found=False,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3),
            cost_usd=cost_usd,
            retries=0,
            error=f"JSON parse error: {e}",
            raw_response=raw,
        )
