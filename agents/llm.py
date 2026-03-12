import os
from functools import lru_cache
from langchain_core.language_models import BaseLanguageModel


@lru_cache(maxsize=1)
def get_llm() -> BaseLanguageModel:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0.3)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.3)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
