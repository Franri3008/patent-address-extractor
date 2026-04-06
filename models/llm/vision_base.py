from abc import ABC, abstractmethod

from PIL.Image import Image

from models.llm.base import LLMResult


class VisionLLMModel(ABC):
    """Abstract interface for vision-capable LLM backends.

    Used in pipeline_mode=1: receives a single page image and extracts
    structured addresses directly, bypassing the OCR stage.

    To add a new vision provider:
      1. Create a new file in models/llm/
      2. Subclass VisionLLMModel and implement extract_addresses_from_image()
      3. Register in models/llm/__init__.py
    """

    @abstractmethod
    def extract_addresses_from_image(self, image: Image, prompt_template: str, page_number: int) -> LLMResult:
        """Extract structured addresses from a single page image."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
