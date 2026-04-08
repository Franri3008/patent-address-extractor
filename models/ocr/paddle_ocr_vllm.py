"""PaddleOCR-VL adapter — served via vLLM (OpenAI-compatible API).

Instead of loading the model locally, this adapter talks to a running
vLLM server that hosts PaddlePaddle/PaddleOCR-VL.

Start the server with:

    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --max-num-batched-tokens 16384 \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 0

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
    """Encode a PIL image as a base64 data URL for the OpenAI vision API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class PaddleOCRVLLMModel(OCRModel):
    """Adapter for PaddleOCR-VL running on a vLLM server."""

    def __init__(self, config: dict) -> None:
        ocr_cfg = config["ocr"]
        self._max_tokens = ocr_cfg.get("max_tokens", 4096);
        self._base_url = ocr_cfg.get("vllm_base_url", _DEFAULT_BASE_URL)
        self._model_name = ocr_cfg.get("vllm_model", _DEFAULT_MODEL)
        self._client = None

    @property
    def model_name(self) -> str:
        return "paddle_ocr_vl_vllm"

    def load(self) -> None:
        """Initialise the OpenAI client pointing at the vLLM server."""
        from openai import OpenAI

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=self._base_url,
            timeout=3600,
        )
        logger.info(
            f"vLLM client ready — server={self._base_url}, model={self._model_name}"
        )

    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images via the vLLM server."""
        if self._client is None:
            raise RuntimeError("Call load() before run().")

        t0 = time.perf_counter()
        texts: list[str] = []

        for img in images:
            text = self._run_inference(img)
            texts.append(text)

        elapsed = time.perf_counter() - t0
        full_text = "\n".join(texts)

        return OCRResult(
            text=full_text,
            elapsed_s=round(elapsed, 3),
            model_name=self.model_name,
            pages_processed=len(images),
        )

    def _run_inference(self, img: PILImage.Image) -> str:
        data_url = _image_to_data_url(img)

        response = self._client.chat.completions.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            messages=[
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
            ],
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()
        logger.info(f"vLLM response: {len(text)} chars")
        return text
