from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # LLM / vLLM
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "dummy"
    model_name: str = "your-model-name"
    llm_max_output_tokens: int | None = None
    llm_request_timeout: float = 120.0
    llm_client_max_retries: int = 2
    llm_timeout_retries: int = 2

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    pages_dir: Path = project_root / "data" / "pages"
    pages_linked_dir: Path = project_root / "data" / "pages_linked"
    obsidian_persons_dir: Path = project_root / "data" / "obsidian" / "persons"
    obsidian_items_dir: Path = project_root / "data" / "obsidian" / "items"

    model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            # при необходимости:
            # extra="ignore",
            # case_sensitive=False,
        )

settings = Settings()
