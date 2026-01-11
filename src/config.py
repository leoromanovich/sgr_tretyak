from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # LLM Provider: "local" (vLLM) или "openrouter"
    llm_provider: str = "local"

    # Local vLLM settings
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "dummy"
    model_name: str = "your-model-name"

    # OpenRouter settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # Common LLM settings
    llm_max_output_tokens: int | None = None
    llm_request_timeout: float = 120.0
    llm_client_max_retries: int = 2
    llm_timeout_retries: int = 2

    # Feature flags
    enable_year_extraction: bool = False
    enable_ner_merge: bool = True  # Добавлять ли NER кандидатов к LLM кандидатам

    @property
    def effective_base_url(self) -> str:
        """Возвращает базовый URL в зависимости от провайдера."""
        if self.llm_provider == "openrouter":
            return self.openrouter_base_url
        return self.openai_base_url

    @property
    def effective_api_key(self) -> str:
        """Возвращает API ключ в зависимости от провайдера."""
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key

    @property
    def effective_model(self) -> str:
        """Возвращает название модели в зависимости от провайдера."""
        if self.llm_provider == "openrouter":
            return self.openrouter_model
        return self.model_name

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    pages_dir: Path = project_root / "data" / "pages"
    pages_linked_dir: Path = project_root / "data" / "pages_linked"
    obsidian_persons_dir: Path = project_root / "data" / "obsidian" / "persons"
    obsidian_items_dir: Path = project_root / "data" / "obsidian" / "items"
    cache_dir: Path = project_root / "cache"

    @property
    def ner_cache_dir(self) -> Path:
        """Директория для кэша NER предсказаний (внутри cache_dir)."""
        return self.cache_dir / "ner_predicts"

    model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            # при необходимости:
            # extra="ignore",
            # case_sensitive=False,
        )

settings = Settings()
