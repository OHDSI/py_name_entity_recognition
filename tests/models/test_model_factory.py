import pytest
from unittest.mock import patch
from py_name_entity_recognition.models.config import ModelConfig
from py_name_entity_recognition.models.factory import ModelFactory


# -----------------------------
# OpenAI Tests
# -----------------------------
@patch("py_name_entity_recognition.models.factory.ChatOpenAI")
def test_create_openai_model(mock_chat_openai, mocker):
    """Tests creating a ChatOpenAI model with defaults."""
    mocker.patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test_key", "PROVIDER": "openai"},
    )
    config = ModelConfig(
        provider="openai",
        AZURE_OPENAI_DEPLOYMENT="gpt-4o",  # alias instead of model_name
    )
    ModelFactory.create(config)
    mock_chat_openai.assert_called_with(model="gpt-4o", temperature=0.0)


@patch("py_name_entity_recognition.models.factory.ChatOpenAI")
def test_create_openai_model_with_extra_params(mock_chat_openai, mocker):
    """Tests creating a ChatOpenAI model with max_tokens and top_p."""
    mocker.patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test_key", "PROVIDER": "openai"},
    )
    config = ModelConfig(
        provider="openai",
        AZURE_OPENAI_DEPLOYMENT="gpt-4o",
        max_tokens=100,
        top_p=0.9,
    )
    ModelFactory.create(config)
    mock_chat_openai.assert_called_with(
        model="gpt-4o", temperature=0.0, max_tokens=100, model_kwargs={"top_p": 0.9}
    )


# -----------------------------
# Anthropic Tests
# -----------------------------
@patch("py_name_entity_recognition.models.factory.ChatAnthropic")
def test_create_anthropic_model(mock_chat_anthropic, mocker):
    """Tests creating a ChatAnthropic model with defaults."""
    mocker.patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "test_key", "PROVIDER": "anthropic"},
    )
    config = ModelConfig(
        provider="anthropic",
        AZURE_OPENAI_DEPLOYMENT="claude-3-opus",
    )
    ModelFactory.create(config)
    mock_chat_anthropic.assert_called_with(model="claude-3-opus", temperature=0.0)


@patch("py_name_entity_recognition.models.factory.ChatAnthropic")
def test_create_anthropic_model_with_extra_params(mock_chat_anthropic, mocker):
    """Tests creating a ChatAnthropic model with max_tokens and top_p."""
    mocker.patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "test_key", "PROVIDER": "anthropic"},
    )
    config = ModelConfig(
        provider="anthropic",
        AZURE_OPENAI_DEPLOYMENT="claude-3-opus",
        max_tokens=150,
        top_p=0.85,
    )
    ModelFactory.create(config)
    mock_chat_anthropic.assert_called_with(
        model="claude-3-opus", temperature=0.0, max_tokens=150, top_p=0.85
    )


# -----------------------------
# Ollama Tests
# -----------------------------
@patch("py_name_entity_recognition.models.factory.ChatOllama")
def test_create_ollama_model(mock_chat_ollama, mocker):
    """Tests creating a ChatOllama model with defaults."""
    mocker.patch.dict(
        "os.environ",
        {"PROVIDER": "ollama"},
    )
    config = ModelConfig(
        provider="ollama",
        AZURE_OPENAI_DEPLOYMENT="llama3",
    )
    ModelFactory.create(config)
    mock_chat_ollama.assert_called_with(model="llama3", temperature=0.0)


@patch("py_name_entity_recognition.models.factory.ChatOllama")
def test_create_ollama_model_with_extra_params(mock_chat_ollama, mocker):
    """Tests creating a ChatOllama model with top_p."""
    mocker.patch.dict(
        "os.environ",
        {"PROVIDER": "ollama"},
    )
    config = ModelConfig(
        provider="ollama",
        AZURE_OPENAI_DEPLOYMENT="llama3",
        top_p=0.95,
    )
    ModelFactory.create(config)
    mock_chat_ollama.assert_called_with(model="llama3", temperature=0.0, top_p=0.95)

'''
# -----------------------------
# Unsupported Provider Test
# -----------------------------
def test_factory_unsupported_provider_raises_error(mocker):
    """Tests that an unsupported provider raises a ValueError."""
    mocker.patch.dict(
        "os.environ",
        {"PROVIDER": "invalid", "AZURE_OPENAI_DEPLOYMENT": "dummy"},
    )
    config = ModelConfig(provider="invalid", AZURE_OPENAI_DEPLOYMENT="dummy")
    with pytest.raises(ValueError, match="Unsupported provider"):
        ModelFactory.create(config)
'''
def test_factory_unsupported_provider_raises_error(mocker):
    """Tests that the factory raises a ValueError for an unsupported provider."""
    # Start with a valid config
    config = ModelConfig(provider="openai", AZURE_OPENAI_DEPLOYMENT="dummy")

    # Patch the provider to an invalid value AFTER validation
    mocker.patch.object(config, "provider", "invalid")

    with pytest.raises(ValueError, match="Unsupported model provider"):
        ModelFactory.create(config)

