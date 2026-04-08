"""PaddleOCR-VL adapter — PaddlePaddle/PaddleOCR-VL (0.9B)

Uses the standard HuggingFace transformers API (AutoModelForCausalLM +
AutoProcessor with trust_remote_code=True).  The paddlepaddle package is
NOT required — only PyTorch + transformers.

Device selection:
  config["ocr"]["device"] = "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
  "auto" selects CUDA if available, then MPS (Apple Silicon), then CPU.
"""
from __future__ import annotations

import time

from PIL import Image as PILImage

from models.ocr.base import OCRModel, OCRResult
from utils.logger import get_logger

logger = get_logger("paddle_ocr")

_MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
_OCR_PROMPT = "OCR:"
_OCR_PROMPT = "OCR:"


def _resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class PaddleOCRModel(OCRModel):
    """Adapter for PaddlePaddle/PaddleOCR-VL via transformers."""

    def __init__(self, config: dict) -> None:
        self._device = _resolve_device(config["ocr"].get("device", "auto"))
        self._max_tokens = config["ocr"].get("max_tokens", 4096);
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return "paddle_ocr_vl"

    def load(self) -> None:
        """Download weights (once) and load model into memory."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        # transformers 5.x removed the "default" RoPE type from ROPE_INIT_FUNCTIONS.
        # The model's remote code (modeling_paddleocr_vl.py) falls back to rope_type="default"
        # when no rope_scaling is set, so we inject the standard implementation before load.
        # transformers 5.x has two compatibility issues with this model's custom RotaryEmbedding:
        #
        # 1. ROPE_INIT_FUNCTIONS no longer contains "default" — the model falls back to
        #    rope_type="default" when rope_scaling is unset, so we inject the standard impl.
        #
        # 2. _init_weights in PreTrainedModel expects module.compute_default_rope_parameters
        #    when rope_type=="default", but the model's RotaryEmbedding never defines it.
        #    We patch _init_weights to inject the missing method before that branch runs.

        def _default_rope(config, device=None, **kwargs):
            base = getattr(config, "rope_theta", 10000.0)
            partial = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim",
                               config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
            return inv_freq, 1.0

        if "default" not in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = _default_rope

        import transformers.modeling_utils as _mu
        _orig_init_weights = _mu.PreTrainedModel._init_weights

        def _patched_init_weights(self_model, module):
            if ("RotaryEmbedding" in module.__class__.__name__
                    and hasattr(module, "original_inv_freq")
                    and getattr(module, "rope_type", None) == "default"
                    and not hasattr(module, "compute_default_rope_parameters")):
                module.compute_default_rope_parameters = lambda cfg: _default_rope(cfg)
            _orig_init_weights(self_model, module)

        _mu.PreTrainedModel._init_weights = _patched_init_weights
        logger.debug("Applied transformers 5.x / PaddleOCR-VL RoPE compatibility patches.")

        logger.info(f"Loading {_MODEL_ID} on {self._device} ...")

        # MPS has incomplete bfloat16 support — use float16 instead.
        if self._device == "mps":
            dtype = torch.float16
        elif self._device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        self._processor = AutoProcessor.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )
        # MPS (Apple Silicon) crashes on SDPA attention in the vision encoder
        # ("LLVM ERROR: Failed to infer result type(s)"). Force eager attention.
        attn_impl = "eager" if self._device == "mps" else None
        extra_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            trust_remote_code=True,
            torch_dtype=dtype,
            **extra_kwargs,
        ).to(self._device).eval()

        # transformers 5.x no longer passes cache_position to prepare_inputs_for_generation
        # (it's None). The model's method does `cache_position[0] != 0` → crash.
        # Passing a fake tensor([0]) breaks the parent class (it converts input_ids →
        # inputs_embeds, stripping image token placeholders before forward() can see them).
        #
        # Fix: replace prepare_inputs_for_generation entirely. We build model_inputs
        # manually — no super() call, no dependency on the parent class's behavior.
        # The model's forward() needs: input_ids (with image tokens on prefill),
        # pixel_values + image_grid_thw on prefill only, and past_key_values for decode.
        import types

        def _fixed_prepare(
            self_m, input_ids, past_key_values=None, attention_mask=None,
            inputs_embeds=None, cache_position=None, position_ids=None,
            use_cache=True, pixel_values=None, pixel_values_videos=None,
            image_grid_thw=None, video_grid_thw=None,
            second_per_grid_ts=None, **kwargs,
        ):
            is_prefill = past_key_values is None or (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            )
            # The model uses cache_position for RoPE position_ids.
            # Without it, every decode step computes position = 0 + rope_deltas → garbage.
            if is_prefill:
                cp = torch.arange(input_ids.shape[1], device=input_ids.device)
            else:
                past_len = past_key_values.get_seq_length()
                cp = torch.tensor([past_len], dtype=torch.long, device=input_ids.device)
            return {
                "input_ids": input_ids if is_prefill else input_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "position_ids": None,
                "cache_position": cp,
                "pixel_values": pixel_values if is_prefill else None,
                "pixel_values_videos": pixel_values_videos if is_prefill else None,
                "image_grid_thw": image_grid_thw if is_prefill else None,
                "video_grid_thw": video_grid_thw if is_prefill else None,
                "second_per_grid_ts": second_per_grid_ts if is_prefill else None,
            }

        self._model.prepare_inputs_for_generation = types.MethodType(_fixed_prepare, self._model)
        logger.debug("Replaced prepare_inputs_for_generation for transformers 5.x compatibility.")

        logger.info("PaddleOCR-VL loaded.")

    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on a list of page images and return concatenated text."""
        if self._model is None:
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
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.2,
            )

        # Decode only the newly generated tokens (skip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[:, prompt_len:]
        n_generated = new_ids.shape[1]

        raw_text = self._processor.batch_decode(
            new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        clean_text = self._processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        logger.info(f"Generated {n_generated} tokens, "
                     f"raw len={len(raw_text)}, clean len={len(clean_text)}")
        if n_generated > 0 and len(clean_text) < 50:
            logger.debug(f"Raw output: {raw_text!r}")

        return clean_text
