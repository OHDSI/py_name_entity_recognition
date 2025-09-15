from unittest.mock import patch

import pytest
from pydantic import ValidationError

from py_name_entity_recognition.models.config import ModelConfig
from py_name_entity_recognition.models.factory import ModelFactory


@patch("py_name_entity_recognition.models.factory.ChatOpenAI")
def test_create_openai_model_with_extra_params(mock_chat_openai, mocker):
    """Tests creating a ChatOpenAI model with max_tokens and top_p."""
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    config = ModelConfig(
        provider="openai", model_name="gpt-4o", max_tokens=100, top_p=0.9
    )
    ModelFactory.create(config)
    mock_chat_openai.assert_called_with(
        model="gpt-4o", temperature=0.0, max_tokens=100, model_kwargs={"top_p": 0.9}
    )


@patch("py_name_entity_recognition.models.factory.ChatOpenAI")
def test_create_openai_model(mock_chat_openai, mocker):
    """Tests the creation of a ChatOpenAI model."""
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    config = ModelConfig(provider="openai", model_name="gpt-4o")
    ModelFactory.create(config)
    mock_chat_openai.assert_called_once()


@patch("py_name_entity_recognition.models.factory.ChatAnthropic")
def test_create_anthropic_model(mock_chat_anthropic, mocker):
    """Tests the creation of a ChatAnthropic model."""
    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
    config = ModelConfig(provider="anthropic", model_name="claude-3-opus")
    ModelFactory.create(config)
    mock_chat_anthropic.assert_called_once()


@patch("py_name_entity_recognition.models.factory.ChatOllama")
def test_create_ollama_model(mock_chat_ollama):
    """Tests the creation of a ChatOllama model."""
    config = ModelConfig(provider="ollama", model_name="llama3")
    ModelFactory.create(config)
    mock_chat_ollama.assert_called_once()


@patch("py_name_entity_recognition.models.factory.AzureChatOpenAI")
def test_create_azure_model(mock_azure_chat_openai, mocker):
    """Tests the creation of an AzureChatOpenAI model."""
    mocker.patch.dict(
        "os.environ",
        {"AZURE_OPENAI_API_KEY": "test_key", "OPENAI_API_VERSION": "2023-05-15"},
    )
    config = ModelConfig(
        provider="azure",
        azure_deployment="my-deployment",
        azure_endpoint="https://my-endpoint.openai.azure.com/",
        api_version="2023-05-15",
    )
    ModelFactory.create(config)
    mock_azure_chat_openai.assert_called_once()


@patch("py_name_entity_recognition.models.factory.AzureChatOpenAI")
def test_create_azure_model_with_extra_params(mock_azure_chat_openai, mocker):
    """Tests creating an AzureChatOpenAI model with max_tokens and top_p."""
    mocker.patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test_key"})
    config = ModelConfig(
        provider="azure",
        azure_deployment="my-deployment",
        azure_endpoint="https://my-endpoint.openai.azure.com/",
        max_tokens=200,
        top_p=0.8,
    )
    ModelFactory.create(config)
    mock_azure_chat_openai.assert_called_with(
        azure_deployment="my-deployment",
        azure_endpoint="https://my-endpoint.openai.azure.com/",
        temperature=0.0,
        max_tokens=200,
        model_kwargs={"top_p": 0.8},
    )


@patch("py_name_entity_recognition.models.factory.ChatAnthropic")
def test_create_anthropic_model_with_extra_params(mock_chat_anthropic, mocker):
    """Tests creating a ChatAnthropic model with max_tokens and top_p."""
    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
    config = ModelConfig(
        provider="anthropic", model_name="claude-3-opus", max_tokens=150, top_p=0.85
    )
    ModelFactory.create(config)
    mock_chat_anthropic.assert_called_with(
        model="claude-3-opus", temperature=0.0, max_tokens=150, top_p=0.85
    )


@patch("py_name_entity_recognition.models.factory.ChatOllama")
def test_create_ollama_model_with_extra_params(mock_chat_ollama):
    """Tests creating a ChatOllama model with top_p."""
    config = ModelConfig(provider="ollama", model_name="llama3", top_p=0.95)
    ModelFactory.create(config)
    mock_chat_ollama.assert_called_with(model="llama3", temperature=0.0, top_p=0.95)


def test_create_azure_model_missing_deployment_raises_error():
    """Tests that missing azure_deployment for Azure raises ValueError."""
    config = ModelConfig(
        provider="azure", azure_endpoint="https://my-endpoint.openai.azure.com/"
    )
    with pytest.raises(
        ValueError, match="'azure_deployment' and 'azure_endpoint' must be specified"
    ):
        ModelFactory.create(config)


def test_unsupported_provider_config_raises_error():
    """Tests that an unsupported provider raises a ValidationError on ModelConfig."""
    with pytest.raises(ValidationError):
        ModelConfig(provider="unsupported_provider")


def test_factory_unsupported_provider_raises_error(mocker):
    """Tests that the factory raises a ValueError for an unsupported provider."""
    config = ModelConfig()
    mocker.patch.object(config, "provider", "unsupported_provider")
    with pytest.raises(
        ValueError, match="Unsupported model provider: 'unsupported_provider'"
    ):
        ModelFactory.create(config)
