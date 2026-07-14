from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///data/news.db"
    news_api_key: Optional[str] = None
    scheduler_interval_minutes: int = 30
    api_key_secret: str = "default-secret-key"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
