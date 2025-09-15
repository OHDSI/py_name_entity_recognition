from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from py_name_entity_recognition.models.config import ModelConfig

# Load environment variables from a .env file if it exists
load_dotenv()


class ModelFactory:
    """
    A factory class for creating and configuring language model instances.
    """

    @staticmethod
    def create(config: ModelConfig) -> BaseLanguageModel:
        """
        Create a new language model instance from a configuration.
        """
        provider_map = {
            "openai": ModelFactory._create_openai,
            "azure": ModelFactory._create_azure,
            "anthropic": ModelFactory._create_anthropic,
            "ollama": ModelFactory._create_ollama,
        }
        creator = provider_map.get(config.provider.lower())
        if not creator:
            raise ValueError(f"Unsupported model provider: '{config.provider}'")
        return creator(config)

    @staticmethod
    def _create_openai(config: ModelConfig) -> ChatOpenAI:
        params: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        model_kwargs: dict[str, Any] = {}
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            model_kwargs["top_p"] = config.top_p
        if model_kwargs:
            params["model_kwargs"] = model_kwargs
        return ChatOpenAI(**params)

    @staticmethod
    def _create_azure(config: ModelConfig) -> AzureChatOpenAI:
        if not all([config.azure_deployment, config.azure_endpoint]):
            raise ValueError(
                "For 'azure' provider, 'azure_deployment' and 'azure_endpoint' must be specified in ModelConfig."
            )
        params: dict[str, Any] = {
            "azure_deployment": config.azure_deployment,
            "azure_endpoint": config.azure_endpoint,
            "temperature": config.temperature,
        }
        model_kwargs: dict[str, Any] = {}
        if config.api_version:
            params["api_version"] = config.api_version
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            model_kwargs["top_p"] = config.top_p
        if model_kwargs:
            params["model_kwargs"] = model_kwargs
        return AzureChatOpenAI(**params)

    @staticmethod
    def _create_anthropic(config: ModelConfig) -> ChatAnthropic:
        params: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p
        return ChatAnthropic(**params)

    @staticmethod
    def _create_ollama(config: ModelConfig) -> ChatOllama:
        params: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.top_p is not None:
            params["top_p"] = config.top_p
        return ChatOllama(**params)
