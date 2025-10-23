"""
Application configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    APP_NAME: str = "Vectorless RAG"
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")

    # Database settings
    MONGODB_URL: str = Field(..., env="MONGODB_URL")
    MONGODB_DATABASE: str = Field(default="vectorless_rag", env="MONGODB_DATABASE")
    MONGODB_MIN_POOL_SIZE: int = Field(default=10, env="MONGODB_MIN_POOL_SIZE")
    MONGODB_MAX_POOL_SIZE: int = Field(default=100, env="MONGODB_MAX_POOL_SIZE")

    # Redis settings
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")

    # MinIO settings
    MINIO_ENDPOINT: str = Field(..., env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(..., env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(..., env="MINIO_SECRET_KEY")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    MINIO_BUCKET_DOCUMENTS: str = Field(
        default="documents", env="MINIO_BUCKET_DOCUMENTS"
    )
    MINIO_BUCKET_IMAGES: str = Field(default="images", env="MINIO_BUCKET_IMAGES")
    MINIO_BUCKET_PROCESSED: str = Field(
        default="processed", env="MINIO_BUCKET_PROCESSED"
    )

    # Gemini AI settings
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    GEMINI_MAX_TOKENS: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    GEMINI_TEMPERATURE: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    GEMINI_TIMEOUT: int = Field(default=60, env="GEMINI_TIMEOUT")

    # File processing settings
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    MAX_PAGES_PER_DOCUMENT: int = Field(default=100, env="MAX_PAGES_PER_DOCUMENT")
    SUPPORTED_FORMATS: str = Field(default="pdf", env="SUPPORTED_FORMATS")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["application/pdf"], env="ALLOWED_FILE_TYPES"
    )
    PDF_DPI: int = Field(default=200, env="PDF_DPI")
    IMAGE_FORMAT: str = Field(default="PNG", env="IMAGE_FORMAT")
    IMAGE_QUALITY: int = Field(default=95, env="IMAGE_QUALITY")

    # Processing settings
    MAX_CONCURRENT_PROCESSES: int = Field(default=4, env="MAX_CONCURRENT_PROCESSES")
    PROCESS_TIMEOUT: int = Field(default=300, env="PROCESS_TIMEOUT")  # 5 minutes
    RETRY_ATTEMPTS: int = Field(default=3, env="RETRY_ATTEMPTS")
    RETRY_DELAY: int = Field(default=1, env="RETRY_DELAY")

    # Cache settings
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    CACHE_PREFIX: str = Field(default="vectorless_rag", env="CACHE_PREFIX")

    # Security settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("ALLOWED_FILE_TYPES", pre=True)
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(",")]
        return v

    @validator("DEBUG", pre=True)
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @validator("MINIO_SECURE", pre=True)
    def parse_minio_secure(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
