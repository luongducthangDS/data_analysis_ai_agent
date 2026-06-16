from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM providers
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    google_api_key: str = ""
    hf_token: str = ""
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct:fastest"
    hf_base_url: str = "https://router.huggingface.co/v1"
    llm_provider: str = "auto"  # auto | groq | gemini | hf

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "data-agent"
    langsmith_tracing: bool = False

    # Auth & multi-tenancy
    api_keys: list[str] = Field(default_factory=list)
    allow_no_auth: bool = True  # True for local dev / web UI

    # Rate limiting
    rate_limit: str = "60/minute"

    # Storage
    database_url: str = "sqlite:///./data/sessions.db"
    data_dir: str = "data"
    session_ttl_days: int = 7

    # CORS
    allowed_origins: str = "*"  # comma-separated list or "*"

    # App
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @field_validator("langsmith_tracing", mode="before")
    @classmethod
    def enable_tracing_if_key(cls, v, info):
        # Auto-enable tracing when key is present
        return v

    @property
    def effective_gemini_key(self) -> str:
        return self.gemini_api_key or self.google_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
