"""Simple disk cache for OCR results.

Cache key: {patent_id}_p{page_number}.txt
Stored as plain UTF-8 text files for easy inspection.

Clear the cache directory when the OCR model or image config changes.
"""
from __future__ import annotations

from pathlib import Path

from utils.logger import get_logger

logger = get_logger("ocr_cache")


class OCRCache:
    def __init__(self, cache_dir: str | Path, enabled: bool = True) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, patent_id: str, page: int) -> str | None:
        if not self._enabled:
            return None
        p = self._key_path(patent_id, page)
        if p.exists():
            self._hits += 1
            return p.read_text(encoding="utf-8")
        self._misses += 1
        return None

    def put(self, patent_id: str, page: int, text: str) -> None:
        if not self._enabled:
            return
        self._key_path(patent_id, page).write_text(text, encoding="utf-8")

    def _key_path(self, patent_id: str, page: int) -> Path:
        clean_id = patent_id.replace("-", "")
        return self._dir / f"{clean_id}_p{page}.txt"

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "enabled": self._enabled}
