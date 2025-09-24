from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """
    A unified configuration schema for instantiating language models from various providers.

    This schema standardizes common parameters and provides fields for provider-specific
    settings, ensuring a consistent interface for model creation.
    """

    provider: Literal["openai", "azure", "anthropic", "ollama"] = Field(
        default="openai",
        description="The name of the LLM provider to use (e.g., 'openai', 'azure').",
    )
    model_name: str = Field(
        default="gpt-4o",
        description="The specific model name to use, e.g., 'gpt-4o', 'claude-3-opus-20240229'.",
    )
    temperature: float = Field(
        default=0.0,
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
        description="Controls diversity via nucleus sampling. 0.5 means half of all likelihood-weighted options are considered.",
    )

    # --- Azure Specific Fields ---
    azure_deployment: Optional[str] = Field(
        default=None,
        description="The name of the Azure OpenAI deployment. Required if provider is 'azure'.",
    )
    azure_endpoint: Optional[str] = Field(
        default=None,
        description="The endpoint URL for the Azure OpenAI service. Required if provider is 'azure'.",
    )
    api_version: Optional[str] = Field(
        default=None,
        alias="azure_api_version",
        description="The API version for the Azure OpenAI service. Defaults to the latest.",
    )

    class Config:
        """Pydantic model configuration."""

        populate_by_name = True
        extra = "ignore"
