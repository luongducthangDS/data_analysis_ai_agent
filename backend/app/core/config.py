from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM providers
    gemini_api_key: str = ""
    gemini_models: str = ""  # comma-separated; trống = default list trong llm_service
    google_api_key: str = ""
    # OpenRouter fallback — chuỗi model free thử lần lượt khi Gemini sập.
    openrouter_api_key: str = ""
    openrouter_models: str = ""  # comma-separated; trống = dùng default list trong llm_service
    # Chuỗi fallback khi llm_provider=auto: Gemini → OpenRouter (nhiều model)
    llm_provider: str = "auto"  # auto | gemini | openrouter

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

    # Báo cáo Telegram (scripts/send_telegram_report.py)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

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
