from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from formuloom.constants import LLM_TIMEOUT_SECONDS, MAX_CONCURRENT_LLM_CALLS


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_prefix="FORMULOOM_", env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", description="Logging verbosity.")

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "FORMULOOM_OPENAI_API_KEY"),
        description="OpenAI API key. Optional at load time; offline paths don't need it.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Escape hatch for local/OpenAI-compatible endpoints (e.g. vLLM, Azure proxy).",
    )
    model: str = Field(default="gpt-5.4-mini", description="Default OpenAI model for classify calls.")
    reasoning_effort: str = Field(default="low", description="Reasoning effort requested from the OpenAI API.")
    service_tier: str | None = Field(default=None, description="OpenAI service tier (e.g. 'flex'), if any.")
    max_concurrent_llm_calls: int = Field(
        default=MAX_CONCURRENT_LLM_CALLS,
        description="Global semaphore size bounding concurrent LLM calls.",
    )
    llm_timeout_seconds: int = Field(
        default=LLM_TIMEOUT_SECONDS,
        description="Per-call timeout before an LLM request is treated as hung.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
