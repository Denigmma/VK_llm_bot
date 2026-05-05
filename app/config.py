from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ReasoningEffort = Literal["low", "medium", "high"]
OpenRouterProfile = Literal["free", "pay"]


class Settings(BaseSettings):
    vk_group_token: str = Field(alias="VK_GROUP_TOKEN")
    vk_confirmation_code: str = Field(alias="VK_CONFIRMATION_CODE")
    vk_secret_key: str = Field(alias="VK_SECRET_KEY")

    openrouter_pay_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("API_KEY_OPENROUTER", "OPENROUTER_API_KEY"),
    )
    openrouter_pay_model: str = Field(
        default="openai/gpt-5.4",
        validation_alias=AliasChoices("MODEL", "OPENROUTER_MODEL"),
    )
    openrouter_free_api_key: str | None = Field(
        default=None,
        alias="FREE_API_KEY_OPENROUTER",
    )
    openrouter_free_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b:free",
        alias="FREE_MODEL",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )

    database_url: str = Field(default="sqlite:///./bot.db", alias="DATABASE_URL")

    default_reasoning_enabled: bool = Field(default=False, alias="DEFAULT_REASONING_ENABLED")
    default_reasoning_effort: ReasoningEffort = Field(default="medium", alias="DEFAULT_REASONING_EFFORT")
    default_max_context_messages: int = Field(default=20, alias="DEFAULT_MAX_CONTEXT_MESSAGES")
    default_temperature: float = Field(default=0.7, alias="DEFAULT_TEMPERATURE")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def openrouter_chat_completions_url(self) -> str:
        return f"{self.openrouter_base_url.rstrip('/')}/chat/completions"

    def get_openrouter_api_key(self, profile: OpenRouterProfile) -> str:
        if profile == "pay":
            api_key = self.openrouter_pay_api_key
            env_name = "API_KEY_OPENROUTER"
        else:
            api_key = self.openrouter_free_api_key
            env_name = "FREE_API_KEY_OPENROUTER"

        if not api_key:
            raise ValueError(f"Профиль {profile} не настроен: переменная {env_name} пустая.")

        return api_key

    def get_openrouter_model(self, profile: OpenRouterProfile) -> str:
        if profile == "pay":
            return self.openrouter_pay_model
        return self.openrouter_free_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
