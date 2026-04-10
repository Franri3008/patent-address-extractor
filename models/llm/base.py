from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# JSON schema for structured output enforcement.
# Backends that support native schema enforcement (Ollama, OpenAI, Google)
# pass this to their API; others rely on prompt instructions.
#
# New format: addresses are deduplicated into an "addresses" array and
# referenced by integer ID inside entities, saving output tokens when
# the same address appears for multiple inventors/applicants.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "addresses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["id", "address"],
            },
        },
        "entities": {
            "type": "object",
            "properties": {
                "inventors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address_id": {"type": ["integer", "null"]},
                        },
                        "required": ["name", "address_id"],
                    },
                },
                "applicants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address_id": {"type": ["integer", "null"]},
                        },
                        "required": ["name", "address_id"],
                    },
                },
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address_id": {"type": ["integer", "null"]},
                        },
                        "required": ["name", "address_id"],
                    },
                },
            },
            "required": ["inventors", "applicants", "agents"],
            "additionalProperties": False,
        },
        "found": {"type": "boolean"},
    },
    "required": ["addresses", "entities", "found"],
    "additionalProperties": False,
};

# Schema used by the vision LLM worker (ollama_vision.py), which processes
# pages one-by-one and needs sections_detected for its stop logic.
VISION_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "inventors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "applicants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                },
                "required": ["name", "address"],
            },
        },
        "sections_detected": {
            "type": "array",
            "items": {"type": "string"},
        },
        "found": {"type": "boolean"},
    },
    "required": ["inventors", "applicants", "agents", "sections_detected", "found"],
    "additionalProperties": False,
};


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

    def reload(self) -> None:
        """Unload and reload the model to reclaim newly available GPU memory.

        Called after the OCR stage frees VRAM so the LLM can use the full GPU.
        Default is a no-op; subclasses should override when applicable.
        """

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
