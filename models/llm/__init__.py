from __future__ import annotations
from models.llm.base import LLMModel


def get_llm_model(config: dict) -> LLMModel:
    """Return an initialised LLM model based on config['llm']['provider']."""
    provider = config["llm"]["provider"];

    if provider == "ollama":
        from models.llm.ollama import OllamaModel
        return OllamaModel(config);
    if provider == "vllm":
        from models.llm.vllm_api import VLLMModel
        return VLLMModel(config);
    if provider == "openai":
        from models.llm.openai_api import OpenAIModel
        return OpenAIModel(config);
    if provider == "anthropic":
        from models.llm.anthropic_api import AnthropicModel
        return AnthropicModel(config);
    if provider == "google":
        from models.llm.google_api import GoogleModel
        return GoogleModel(config);

    raise ValueError(f"Unknown LLM provider: '{provider}'. Add it to models/llm/__init__.py.");


def get_vision_llm_model(config: dict):
    """Return an initialised vision LLM model based on config['vision_llm']['provider']."""
    provider = config["vision_llm"]["provider"];

    if provider == "ollama":
        from models.llm.ollama_vision import OllamaVisionModel
        return OllamaVisionModel(config);

    raise ValueError(f"Unknown vision LLM provider: '{provider}'. Add it to models/llm/__init__.py.");
