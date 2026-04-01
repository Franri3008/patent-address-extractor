"""dots.ocr-1.5 adapter — rednote-hilab/dots.ocr-1.5

Installation:
  pip install modelscope torch torchvision transformers
  # Model weights are downloaded automatically on first load (~several GB).
  # Alternatively, clone from GitHub:
  #   https://github.com/rednote-hilab/dots.ocr

Device selection:
  config["ocr"]["device"] = "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
  "auto" selects CUDA if available, then MPS (Apple Silicon), then CPU.

TODO: Confirm the exact model API after installation by checking:
  https://modelscope.cn/models/rednote-hilab/dots.ocr-1.5
  and update _run_inference() accordingly if the interface differs.
"""
from __future__ import annotations

import time

from PIL import Image as PILImage

from models.ocr.base import OCRModel, OCRResult
from utils.logger import get_logger

logger = get_logger("dots_ocr");


def _resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg;
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda";
        if torch.backends.mps.is_available():
            return "mps";
    except ImportError:
        pass;
    return "cpu";


class DotsOCRModel(OCRModel):
    """Adapter for rednote-hilab/dots.ocr-1.5."""

    _MODEL_ID = "rednote-hilab/dots.ocr-1.5";

    def __init__(self, config: dict) -> None:
        self._device = _resolve_device(config["ocr"].get("device", "auto"));
        self._model = None;
        self._processor = None;

    @property
    def model_name(self) -> str:
        return "dots_ocr_1.5";

    def load(self) -> None:
        """Load model weights. Called once at startup."""
        logger.info(f"Loading {self._MODEL_ID} on {self._device} ...");
        try:
            # Primary: ModelScope pipeline
            from modelscope.pipelines import pipeline as ms_pipeline
            from modelscope.utils.constant import Tasks
            self._pipe = ms_pipeline(
                task=Tasks.ocr_recognition,
                model=self._MODEL_ID,
                device=self._device,
            );
            self._backend = "modelscope";
        except Exception:
            # Fallback: HuggingFace transformers
            # TODO: replace with actual HF class if dots.ocr exposes one
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            self._processor = AutoProcessor.from_pretrained(self._MODEL_ID);
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._MODEL_ID,
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            ).to(self._device);
            self._backend = "transformers";

        logger.info(f"dots.ocr-1.5 loaded (backend={self._backend}).");

    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images and return concatenated text."""
        if self._model is None and not hasattr(self, "_pipe"):
            raise RuntimeError("Call load() before run().");

        t0 = time.perf_counter();
        texts: list[str] = [];

        for img in images:
            text = self._run_inference(img);
            texts.append(text);

        elapsed = time.perf_counter() - t0;
        full_text = "\n".join(texts);

        return OCRResult(
            text=full_text,
            elapsed_s=round(elapsed, 3),
            model_name=self.model_name,
            pages_processed=len(images),
        );

    def _run_inference(self, img: PILImage.Image) -> str:
        """Run inference on a single image and return OCR text."""
        if self._backend == "modelscope":
            result = self._pipe(img);
            if isinstance(result, dict):
                return result.get("text", str(result));
            return str(result);

        # HuggingFace transformers fallback
        # TODO: adjust inputs/outputs once model HF interface is confirmed
        import torch
        inputs = self._processor(images=img, return_tensors="pt").to(self._device);
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=4096);
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0];
