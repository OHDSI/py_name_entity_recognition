import logging
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from datasets import Dataset
from pydantic import BaseModel, Field

from py_name_entity_recognition.data_handling.io import (
    _resolve_schema,
    _yield_texts,
    biores_to_entities,
    extract_entities,
)
from py_name_entity_recognition.schemas.core_schemas import BaseEntity, Entities


class Person(BaseEntity):
    pass


class Location(BaseEntity):
    pass


class MySchema(BaseModel):
    name: str = Field(...)


def test_resolve_schema_invalid_key(caplog):
    with caplog.at_level(logging.WARNING):
        _resolve_schema({"preset": "CLINICAL_TRIAL_CORE", "invalid_key": "value"})
    assert "Invalid keys found and ignored" in caplog.text


def test_resolve_schema_empty_config():
    with pytest.raises(ValueError, match="Invalid or empty configuration"):
        _resolve_schema({})


def test_resolve_schema_invalid_type():
    with pytest.raises(TypeError, match="Invalid schema input type"):
        _resolve_schema(123)


@pytest.mark.parametrize(
    "tagged_tokens, expected_entities",
    [
        (
            [("John", "S-PERSON"), ("works", "O"), ("at", "O"), ("Google", "S-ORG")],
            Entities(
                entities=[
                    BaseEntity(type="PERSON", text="John"),
                    BaseEntity(type="ORG", text="Google"),
                ]
            ),
        ),
        (
            [("New", "B-LOCATION"), ("York", "E-LOCATION")],
            Entities(entities=[BaseEntity(type="LOCATION", text="New York")]),
        ),
        (
            [("San", "B-LOCATION"), ("Francisco", "I-LOCATION"), ("Bay", "E-LOCATION")],
            Entities(entities=[BaseEntity(type="LOCATION", text="San Francisco Bay")]),
        ),
        ([], Entities(entities=[])),
        (
            [("New", "B-LOCATION"), ("York", "B-LOCATION")],
            Entities(
                entities=[
                    BaseEntity(type="LOCATION", text="New"),
                    BaseEntity(type="LOCATION", text="York"),
                ]
            ),
        ),
    ],
)
def test_biores_to_entities(tagged_tokens, expected_entities):
    assert biores_to_entities(tagged_tokens) == expected_entities


@pytest.mark.asyncio
async def test_yield_texts():
    # Test with string
    assert [x async for x in _yield_texts("hello", None)] == [("hello", 0)]

    # Test with list
    assert [x async for x in _yield_texts(["a", "b"], None)] == [("a", 0), ("b", 1)]

    # Test with DataFrame
    df = pd.DataFrame({"text": ["c", "d"], "other": [1, 2]})
    assert [x async for x in _yield_texts(df, "text")] == [("c", 0), ("d", 1)]

    # Test with Dataset
    ds = Dataset.from_pandas(df)
    assert [x async for x in _yield_texts(ds, "text")] == [("c", 0), ("d", 1)]

    # Test unsupported type
    with pytest.raises(TypeError):
        [x async for x in _yield_texts(123, None)]

    # Test missing text_column for DataFrame
    with pytest.raises(ValueError, match="`text_column` must be specified"):
        [x async for x in _yield_texts(df, None)]

    # Test missing text_column for Dataset
    with pytest.raises(ValueError, match="`text_column` must be specified"):
        [x async for x in _yield_texts(ds, None)]


@pytest.mark.asyncio
@patch("py_name_entity_recognition.data_handling.io.CoreEngine")
@patch("py_name_entity_recognition.data_handling.io.ModelFactory")
async def test_extract_entities(mock_factory, mock_engine):
    mock_engine_instance = mock_engine.return_value
    mock_engine_instance.run = AsyncMock(return_value=[("John", "S-PERSON")])

    # Test with string
    result = await extract_entities("John", Person, output_format="json")
    assert result == {"entities": [{"type": "PERSON", "text": "John"}]}

    # Test with list
    results = await extract_entities(["John", "Doe"], Person, output_format="conll")
    assert len(results) == 2
    assert results[0] == [("John", "S-PERSON")]

    # Test with invalid schema
    with pytest.raises(ValueError):
        await extract_entities("test", "not a schema")

    # Test with invalid output format
    with pytest.raises(ValueError):
        await extract_entities("test", Person, output_format="invalid")


@pytest.mark.asyncio
@patch("py_name_entity_recognition.data_handling.io.CoreEngine")
@patch("py_name_entity_recognition.data_handling.io.ModelFactory")
async def test_extract_entities_with_model_config_dict(mock_factory, mock_engine):
    mock_engine_instance = mock_engine.return_value
    mock_engine_instance.run = AsyncMock(return_value=[("John", "S-PERSON")])

    await extract_entities(
        "John", Person, model_config={"provider": "openai", "name": "gpt-4"}
    )
    mock_factory.create.assert_called_once()


@pytest.mark.asyncio
@patch("py_name_entity_recognition.data_handling.io.CoreEngine")
@patch("py_name_entity_recognition.data_handling.io.ModelFactory")
async def test_extract_entities_with_empty_list(mock_factory, mock_engine):
    result = await extract_entities([], Person)
    assert result == []
