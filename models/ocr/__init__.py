from __future__ import annotations
from models.ocr.base import OCRModel


def get_ocr_model(config: dict) -> OCRModel:
    """Return an initialised OCR model based on config['ocr']['model']."""
    key = config["ocr"]["model"];

    if key == "dots_ocr":
        from models.ocr.dots_ocr import DotsOCRModel
        return DotsOCRModel(config);

    raise ValueError(f"Unknown OCR model: '{key}'. Add it to models/ocr/__init__.py.");
