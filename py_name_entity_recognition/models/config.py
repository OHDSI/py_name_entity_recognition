from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelConfig(BaseSettings):
    """
    A unified configuration schema that automatically loads settings from a .env file
    for instantiating language models from various providers.
    """

    # --- General Fields ---
    # Loads from the PROVIDER variable in your .env file
    provider: Literal["openai", "azure", "anthropic", "ollama"] = Field(
        default="azure",
        alias="PROVIDER",
        description="The name of the LLM provider to use (e.g., 'openai', 'azure').",
    )

    # This is the key change: It now reads 'AZURE_OPENAI_DEPLOYMENT' from .env
    # and makes it a required field.
    model_name: str = Field(
        default=..., # '...' makes this field required
        alias="AZURE_OPENAI_DEPLOYMENT",
        description="The specific model/deployment name to use.",
    )

    # Loads from the TEMPERATURE variable in your .env file
    temperature: float = Field(
        default=0.0,
        alias="TEMPERATURE",
        ge=0.0,
        le=2.0,
        description="Controls randomness. Lower values make the model more deterministic.",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="The maximum number of tokens to generate in the completion.",
    )
    
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling.",
    )

    # --- Azure Specific Fields ---
    # These now also load directly from your .env file using aliases
    azure_deployment: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_DEPLOYMENT",
        description="The name of the Azure OpenAI deployment.",
    )
    
    azure_endpoint: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_ENDPOINT",
        description="The endpoint URL for the Azure OpenAI service.",
    )
    
    api_version: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_API_VERSION",
        description="The API version for the Azure OpenAI service.",
    )

    # This model_config block tells Pydantic to load variables from a .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
