from __future__ import annotations

import base64
import io
import json
import time

from jinja2 import Template
from PIL.Image import Image

from models.llm.base import VISION_EXTRACTION_SCHEMA, LLMResult
from models.llm.vision_base import VisionLLMModel
from utils.logger import get_logger

logger = get_logger("llm.ollama_vision");


class OllamaVisionModel(VisionLLMModel):
    def __init__(self, config: dict) -> None:
        self._model = config["vision_llm"]["model"];
        self._temperature = config["vision_llm"].get("temperature", 0.1);
        self._think = config["vision_llm"].get("think", False);

    @property
    def provider_name(self) -> str:
        return "ollama_vision";

    @property
    def model_name(self) -> str:
        return self._model;

    def extract_addresses_from_image(self, image: Image, prompt_template: str, page_number: int, template_vars: dict | None = None) -> LLMResult:
        import ollama

        prompt = Template(prompt_template).render(page_number=page_number, **(template_vars or {}));

        buf = io.BytesIO();
        image.save(buf, format="JPEG");
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8");

        t0 = time.perf_counter();
        response = ollama.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }],
            format=VISION_EXTRACTION_SCHEMA,
            options={"temperature": self._temperature},
            think=self._think,
        );
        elapsed = time.perf_counter() - t0;

        raw = response["message"]["content"];
        tokens_in = response.get("prompt_eval_count", 0);
        tokens_out = response.get("eval_count", 0);

        return _parse_response(raw, elapsed, tokens_in, tokens_out);


def _parse_response(raw: str, elapsed: float, tokens_in: int, tokens_out: int) -> LLMResult:
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
            cost_usd=None,
            retries=0,
            raw_response=raw,
        );
    except json.JSONDecodeError as e:
        return LLMResult(
            inventors=[], applicants=[], agents=[],
            sections_detected=[], found=False,
            tokens_in=tokens_in, tokens_out=tokens_out,
            elapsed_s=round(elapsed, 3), cost_usd=None, retries=0,
            error=f"JSON parse error: {e}",
            raw_response=raw,
        );
