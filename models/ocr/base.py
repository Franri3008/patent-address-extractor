from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image as PILImage


@dataclass
class OCRResult:
    text: str;
    elapsed_s: float;
    model_name: str;
    pages_processed: int;
    char_count: int = field(init=False);

    def __post_init__(self) -> None:
        self.char_count = len(self.text);


class OCRModel(ABC):
    """Abstract interface for all OCR backends.

    To add a new OCR model:
      1. Create a new file in models/ocr/
      2. Subclass OCRModel and implement load() and run()
      3. Register in models/ocr/__init__.py
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory/GPU. Called once at startup."""
        ...

    @abstractmethod
    def run(self, images: list[PILImage.Image]) -> OCRResult:
        """Run OCR on one or more page images, return concatenated text."""
        ...

    async def run_async(self, images: list[PILImage.Image]) -> OCRResult:
        """Async version. Default falls back to sync run in a thread."""
        import asyncio
        return await asyncio.to_thread(self.run, images)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for metadata."""
        ...
