"""Configuration management for the application.

This module provides a centralized configuration system that loads settings
from environment variables (via .env file) with sensible defaults.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with defaults."""

    model_config = SettingsConfigDict(
        # Look for .env file in the server directory (parent of src)
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations",
    )

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key for vector database operations",
    )
    pinecone_index: str = Field(
        default="default-index",
        description="Pinecone index name",
    )

    # PostgreSQL Configuration
    postgres_db_name: str = Field(
        default="farsight",
        description="PostgreSQL database name",
    )
    postgres_user: str = Field(
        default="postgres",
        description="PostgreSQL username",
        alias="postgres",  # Maps POSTGRES env var to postgres_user
    )
    postgres_password: str = Field(
        default="postgres",
        description="PostgreSQL password",
    )
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host address",
    )
    postgres_port: int = Field(
        default=5432,
        description="PostgreSQL port number",
    )
    
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse Secret Key"
    )
    
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse Public Key"
    )
    
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse Base URL"
    )

    @field_validator("postgres_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate that port is within valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @property
    def postgres_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}"
        )

    @property
    def postgres_async_connection_string(self) -> str:
        """Generate PostgreSQL async connection string for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}"
        )

    @property
    def postgres_config(self) -> dict:
        """Generate PostgreSQL config object for asyncpg."""
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "database": self.postgres_db_name,
        }


# Singleton instance - import this in other modules
settings = Settings()
