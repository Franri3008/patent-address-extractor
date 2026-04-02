"""dots.ocr-1.5 adapter — rednote-hilab/dots.ocr-1.5

Loads the model directly via HuggingFace transformers (AutoModelForCausalLM +
AutoProcessor with trust_remote_code=True), bypassing the modelscope pipeline
which does not register an ocr-recognition handler for this model.

Weights are downloaded via modelscope.snapshot_download on first load (~6 GB).

Device selection:
  config["ocr"]["device"] = "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
  "auto" selects CUDA if available, then MPS (Apple Silicon), then CPU.

Inference strategy — manual greedy decode loop:
  generate() in transformers 5.x calls prepare_inputs_for_generation(), which
  in this model's remote code has a broken vision-injection path that does not
  work correctly under the new API (cache_position / is_first_iteration change).
  Rather than patching cached remote-code files we:
    1. Pre-compute inputs_embeds with vision tokens already merged in via
       model.prepare_inputs_embeds() — the model's own method, called directly.
    2. Run a simple greedy decode loop calling model() directly, with
       inputs_embeds on the first step and cached KV + last token on subsequent
       steps.  generate() is never involved.
  All fixes live in this project file; no cached model files need patching.
"""
from __future__ import annotations

import time

from PIL import Image as PILImage

from models.ocr.base import OCRModel, OCRResult
from utils.logger import get_logger

logger = get_logger("dots_ocr");

_MAX_NEW_TOKENS = 4096


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
        """Download weights (once) and load model into memory."""
        import torch
        from modelscope import snapshot_download
        from transformers import AutoProcessor, AutoModelForCausalLM

        logger.info(f"Loading {self._MODEL_ID} on {self._device} ...");

        model_dir = snapshot_download(self._MODEL_ID);
        logger.info(f"Model dir: {model_dir}");

        self._processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        );
        dtype = torch.bfloat16 if self._device != "cpu" else torch.float32;
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device);
        self._model.eval();

        logger.info("dots.ocr-1.5 loaded.");

    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images and return concatenated text."""
        if self._model is None:
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
        """Run inference on a single image using a manual greedy decode loop.

        Bypasses generate() / prepare_inputs_for_generation entirely to avoid
        the broken vision-injection path in the model's remote code under
        transformers 5.x.  All logic lives here in the project.
        """
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Transcribe all text in this image verbatim."},
                ],
            }
        ];

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        );
        inputs = self._processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self._device);
        inputs.pop("mm_token_type_ids", None);

        input_ids      = inputs["input_ids"];           # [1, seq_len]
        attention_mask = inputs["attention_mask"];      # [1, seq_len]
        pixel_values   = inputs["pixel_values"].to(dtype=self._model.dtype);
        image_grid_thw = inputs["image_grid_thw"];

        eos_id = self._model.config.eos_token_id;
        if isinstance(eos_id, list):
            eos_id = eos_id[0];

        logger.debug(
            f"input_ids={input_ids.shape}  pixel_values={pixel_values.shape}"
            f"  image_grid_thw={image_grid_thw.tolist()}"
        );

        generated: list[int] = [];

        with torch.no_grad():
            # Step 1 — pre-compute vision-merged embeddings.
            img_mask = input_ids == self._model.config.image_token_id;
            logger.debug(f"image_token_id={self._model.config.image_token_id}  img_mask_sum={img_mask.sum().item()}");
            inputs_embeds = self._model.prepare_inputs_embeds(
                input_ids, pixel_values, image_grid_thw, img_mask
            );

            # Step 2 — greedy decode loop.
            # First forward: pass input_ids (satisfies the assertion in the
            # model's forward()) AND our pre-computed inputs_embeds (the model
            # uses inputs_embeds when it is not None, so vision tokens are used).
            # Subsequent forwards: pass only the last generated token + KV cache.
            past_key_values = None;
            cur_mask = attention_mask;

            for step in range(_MAX_NEW_TOKENS):
                if step == 0:
                    outputs = self._model(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=cur_mask,
                        use_cache=True,
                        return_dict=True,
                    );
                else:
                    last_token = torch.tensor(
                        [[generated[-1]]], dtype=torch.long, device=self._device
                    );
                    cur_mask = torch.cat(
                        [cur_mask, torch.ones(1, 1, dtype=torch.long, device=self._device)],
                        dim=1,
                    );
                    outputs = self._model(
                        input_ids=last_token,
                        attention_mask=cur_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    );

                past_key_values = outputs.past_key_values;
                next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item());

                if next_token == eos_id:
                    break;
                generated.append(next_token);

        logger.debug(f"generated {len(generated)} tokens");

        if not generated:
            return "";

        output_tensor = torch.tensor([generated], dtype=torch.long);
        return self._processor.batch_decode(
            output_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0];
