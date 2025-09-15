import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

import pandas as pd
from datasets import Dataset
from pydantic import BaseModel

from py_name_entity_recognition.catalog import get_schema
from py_name_entity_recognition.core.engine import CoreEngine
from py_name_entity_recognition.models.config import ModelConfig
from py_name_entity_recognition.models.factory import ModelFactory
from py_name_entity_recognition.schemas.core_schemas import BaseEntity, Entities

# Configure logging
logger = logging.getLogger(__name__)


def _resolve_schema(
    schema_input: Union[type[BaseModel], str, dict[str, Any]]
) -> type[BaseModel]:
    """
    Helper function to resolve various input types into a Pydantic BaseModel.

    This function enables flexible schema definition by accepting:
    1. A direct Pydantic model class (for backward compatibility).
    2. A string representing a preset name from the catalog.
    3. A dictionary with configuration options for dynamic schema generation.

    Args:
        schema_input: The schema definition to resolve.

    Returns:
        A Pydantic BaseModel class ready for use in extraction.

    Raises:
        TypeError: If the schema_input is of an unsupported type.
        ValueError: If a dictionary configuration for get_schema is invalid.
    """
    if isinstance(schema_input, str):
        # Case 2: Input is a preset name from the catalog
        logger.info(f"Resolving schema from preset: '{schema_input}'")
        return get_schema(preset=schema_input)

    elif isinstance(schema_input, dict):
        # Case 3: Input is a configuration dictionary for get_schema
        logger.info("Resolving schema from configuration dictionary.")

        # Ensure the dictionary keys are valid arguments for get_schema
        valid_keys = inspect.signature(get_schema).parameters.keys()
        config = {k: v for k, v in schema_input.items() if k in valid_keys}

        if len(config) != len(schema_input):
            invalid_keys = set(schema_input.keys()) - set(config.keys())
            logger.warning(
                f"Invalid keys found and ignored in schema configuration: {invalid_keys}"
            )

        if not config:
            raise ValueError(
                "Invalid or empty configuration dictionary provided for schema generation."
            )

        return get_schema(**config)

    elif inspect.isclass(schema_input) and issubclass(schema_input, BaseModel):
        # Case 1: Input is a direct Pydantic model (backward compatibility)
        logger.info("Using provided Pydantic model as schema.")
        return schema_input

    else:
        raise TypeError(
            f"Invalid schema input type: {type(schema_input)}. "
            "Expected a Pydantic BaseModel class, a string (preset name), or a dict (configuration)."
        )


def biores_to_entities(tagged_tokens: list[tuple[str, str]]) -> Entities:
    """
    Converts a list of BIOSES-tagged tokens back into structured entities.
    """
    entities: list[BaseEntity] = []
    current_entity_tokens: list[str] = []
    current_entity_type: Optional[str] = None

    for token, tag in tagged_tokens:
        prefix = tag[0] if tag != "O" else "O"
        entity_type = tag[2:] if len(tag) > 1 else None

        # If we are in an entity and the new tag indicates an end or a new entity
        if current_entity_type and prefix in ("B", "S", "O"):
            entities.append(
                BaseEntity(
                    type=current_entity_type, text=" ".join(current_entity_tokens)
                )
            )
            current_entity_tokens = []
            current_entity_type = None

        # Start a new entity
        if prefix in ("B", "S"):
            current_entity_tokens.append(token)
            current_entity_type = entity_type
        # Continue an existing entity
        elif prefix in ("I", "E") and entity_type == current_entity_type:
            current_entity_tokens.append(token)

        # If the tag is an end or a single-token entity, close it
        if current_entity_type and prefix in ("E", "S"):
            entities.append(
                BaseEntity(
                    type=current_entity_type, text=" ".join(current_entity_tokens)
                )
            )
            current_entity_tokens = []
            current_entity_type = None

    # Add any remaining entity
    if current_entity_tokens and current_entity_type:
        entities.append(
            BaseEntity(type=current_entity_type, text=" ".join(current_entity_tokens))
        )

    return Entities(entities=entities)


async def _yield_texts(
    input_data: Any, text_column: Optional[str]
) -> AsyncGenerator[tuple[str, Any], None]:
    """Asynchronously yields text and context from various input data types."""
    if isinstance(input_data, str):
        yield input_data, 0
        return

    if isinstance(input_data, list):
        for i, text in enumerate(input_data):
            if isinstance(text, str):
                yield text, i
        return

    if isinstance(input_data, pd.DataFrame):
        if not text_column:
            raise ValueError("`text_column` must be specified for DataFrame inputs.")
        if text_column not in input_data.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        for index, row in input_data.iterrows():
            yield row[text_column], index
        return

    if isinstance(input_data, Dataset):
        if not text_column:
            raise ValueError("`text_column` must be specified for Dataset inputs.")
        if text_column not in input_data.column_names:
            raise ValueError(f"Column '{text_column}' not found in Dataset.")
        for i, item in enumerate(input_data):
            yield item[text_column], i
        return

    raise TypeError(f"Unsupported input data type: {type(input_data)}")


async def extract_entities(
    input_data: Any,
    schema: Union[type[BaseModel], str, dict[str, Any]],
    text_column: Optional[str] = None,
    model_config: Optional[Union[dict, ModelConfig]] = None,
    mode: str = "lcel",
    output_format: str = "conll",
) -> Union[list[Any], Any]:
    """
    High-level public API for extracting entities from various input sources.

    Args:
        input_data: The data to process. Can be a string, list of strings,
                    pandas DataFrame, or Hugging Face Dataset.
        schema: The extraction schema definition. Can be:
                1. A Pydantic model class (backward compatibility).
                2. A string representing a preset name from the catalog (e.g., "CLINICAL_TRIAL_CORE").
                3. A dictionary for dynamic schema configuration (passed to `catalog.get_schema`).
        text_column: The name of the column containing text to process (for DataFrames/Datasets).
        model_config: Configuration for the language model.
        mode: The extraction mode to use ('lcel' or 'agentic').
        output_format: The desired output format ('conll' or 'json').

    Returns:
        The extracted entities, formatted as specified. Returns a single result
        for a string input, or a list of results for iterable inputs.
    """
    if model_config is None:
        model_config_obj = ModelConfig()
    elif isinstance(model_config, dict):
        model_config_obj = ModelConfig(**model_config)
    else:
        model_config_obj = model_config

    # Resolve the flexible schema input into a concrete Pydantic model
    resolved_schema = _resolve_schema(schema)

    model = ModelFactory.create(model_config_obj)
    engine = CoreEngine(model=model, schema=resolved_schema)

    texts_to_process = []
    async for text, _ in _yield_texts(input_data, text_column):
        texts_to_process.append(text)

    tasks = [engine.run(text, mode=mode) for text in texts_to_process]
    conll_results: list[list[tuple[str, str]]] = await asyncio.gather(*tasks)

    output_results: Union[list[dict[str, Any]], list[list[tuple[str, str]]]]
    if output_format == "json":
        output_results = [biores_to_entities(r).model_dump() for r in conll_results]
    elif output_format == "conll":
        output_results = conll_results
    else:
        raise ValueError(
            f"Unsupported output format: '{output_format}'. Choose 'conll' or 'json'."
        )

    return output_results[0] if isinstance(input_data, str) else output_results
