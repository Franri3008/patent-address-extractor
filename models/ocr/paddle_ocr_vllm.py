"""PaddleOCR-VL adapter — served via vLLM (OpenAI-compatible API).

Instead of loading the model locally, this adapter talks to a running
vLLM server that hosts PaddlePaddle/PaddleOCR-VL.

Start the server with (tune values based on GPU sharing / benchmarks):

    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --served-model-name PaddleOCR-VL-0.9B \
        --gpu-memory-utilization 0.55 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 8 \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 2

    Recommended sweep for tuning (monitor nvidia-smi for OOM/preemption):
      --gpu-memory-utilization: 0.40, 0.45, 0.50, 0.55
      --max-num-seqs: 4, 8
      --max-num-batched-tokens: 16384, 32768

Config keys (under "ocr"):
    model        : "paddle_ocr_vllm"
    vllm_base_url: "http://localhost:8000/v1"   (default)
    vllm_model   : "PaddlePaddle/PaddleOCR-VL"  (default, must match --model on server)
"""
from __future__ import annotations

import base64
import io
import time

from PIL import Image as PILImage

from models.ocr.base import OCRModel, OCRResult
from utils.logger import get_logger

logger = get_logger("paddle_ocr_vllm")

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL"
_OCR_PROMPT = "OCR:"


def _image_to_data_url(img: PILImage.Image) -> str:
    """Encode a PIL image as a base64 data URL for the OpenAI vision API.

    Uses JPEG encoding for much smaller payloads (~5-10x vs PNG).
    Quality 85 is safe for OCR on text-heavy patent pages.
    """
    buf = io.BytesIO()
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class PaddleOCRVLLMModel(OCRModel):
    """Adapter for PaddleOCR-VL running on a vLLM server."""

    def __init__(self, config: dict) -> None:
        ocr_cfg = config["ocr"]
        self._max_tokens = ocr_cfg.get("max_tokens", 4096)
        self._base_url = ocr_cfg.get("vllm_base_url", _DEFAULT_BASE_URL)
        self._model_name = ocr_cfg.get("vllm_model", _DEFAULT_MODEL)
        self._client = None
        self._async_client = None

    @property
    def model_name(self) -> str:
        return "paddle_ocr_vl_vllm"

    def load(self) -> None:
        """Initialise both sync and async OpenAI clients pointing at the vLLM server."""
        from openai import AsyncOpenAI, OpenAI

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=self._base_url,
            timeout=3600,
        )
        self._async_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=self._base_url,
            timeout=3600,
        )
        logger.info(
            f"vLLM client ready — server={self._base_url}, model={self._model_name}"
        )

    def _build_messages(self, img: PILImage.Image) -> list[dict]:
        data_url = _image_to_data_url(img)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": _OCR_PROMPT,
                    },
                ],
            }
        ]

    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images via the vLLM server (sync)."""
        if self._client is None:
            raise RuntimeError("Call load() before run().")

        t0 = time.perf_counter()
        texts: list[str] = []

        for img in images:
            response = self._client.chat.completions.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                messages=self._build_messages(img),
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"vLLM response: {len(text)} chars")
            texts.append(text)

        elapsed = time.perf_counter() - t0
        full_text = "\n".join(texts)

        return OCRResult(
            text=full_text,
            elapsed_s=round(elapsed, 3),
            model_name=self.model_name,
            pages_processed=len(images),
        )

    async def run_async(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images via the vLLM server (async).

        This enables concurrent requests to vLLM for cross-patent batching.
        """
        if self._async_client is None:
            raise RuntimeError("Call load() before run_async().")

        t0 = time.perf_counter()
        texts: list[str] = []

        for img in images:
            response = await self._async_client.chat.completions.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                messages=self._build_messages(img),
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"vLLM async response: {len(text)} chars")
            texts.append(text)

        elapsed = time.perf_counter() - t0
        full_text = "\n".join(texts)

        return OCRResult(
            text=full_text,
            elapsed_s=round(elapsed, 3),
            model_name=self.model_name,
            pages_processed=len(images),
        )
