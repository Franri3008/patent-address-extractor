from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    inventors: list[dict];         # [{name: str, address: str | None}]
    applicants: list[dict];        # [{name: str, address: str | None}]
    agents: list[dict];            # [{name: str, address: str | None}]
    sections_detected: list[str];  # e.g. ["(71)", "(72)", "(74)"]
    found: bool;
    tokens_in: int;
    tokens_out: int;
    elapsed_s: float;
    cost_usd: float | None;        # None for local models
    retries: int;
    error: str | None = None;
    raw_response: str | None = None;


class LLMModel(ABC):
    """Abstract interface for all LLM address-extraction backends.

    To add a new LLM provider:
      1. Create a new file in models/llm/
      2. Subclass LLMModel and implement extract_addresses()
      3. Register in models/llm/__init__.py
    """

    @abstractmethod
    def extract_addresses(self, ocr_text: str, prompt_template: str, template_vars: dict | None = None) -> LLMResult:
        """Extract structured inventor/applicant/agent addresses from OCR text."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier, e.g. 'ollama', 'openai'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name, e.g. 'gemma3:27b', 'gpt-4o'."""
        ...
