"""
pyNameEntityRecognition: A Python package for state-of-the-art LLM-based Named Entity Recognition.

This package leverages LangChain for LLM orchestration and LangGraph for creating robust,
agentic, and self-refining extraction workflows. It includes a comprehensive catalog
of predefined schemas for scientific and biomedical text.
"""

import os

import toml

from .catalog import PRESETS, get_schema, register_entity
from .data_handling.io import extract_entities

# Correctly locate pyproject.toml relative to the current file
# __file__ -> /app/py_name_entity_recognition/__init__.py
# os.path.dirname(__file__) -> /app/py_name_entity_recognition
# os.path.dirname(os.path.dirname(__file__)) -> /app
pyproject_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
)

with open(pyproject_path) as f:
    pyproject_data = toml.load(f)

__version__ = pyproject_data["tool"]["poetry"]["version"]

__all__ = [
    "extract_entities",
    "get_schema",
    "register_entity",
    "PRESETS",
]
