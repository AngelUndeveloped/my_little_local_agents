from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Gmail API Settings
    gmail_client_id: str
    gmail_client_secret: str
    gmail_refresh_token: str

    # LLM Settings
    llm_api_key: Optional[str] = None
    llm_model_type: str = "gemini"  # or "local"
    local_llm_url: Optional[str] = None

    # Database Settings
    database_url: str = "sqlite:///email_agent.db"

    # Redis MCP Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0

    # Application Settings
    log_level: str = "INFO"
    check_interval: int = 60  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings() 