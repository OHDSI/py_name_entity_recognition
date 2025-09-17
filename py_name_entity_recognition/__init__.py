"""
pyNameEntityRecognition: A Python package for state-of-the-art LLM-based Named Entity Recognition.

This package leverages LangChain for LLM orchestration and LangGraph for creating robust,
agentic, and self-refining extraction workflows. It includes a comprehensive catalog
of predefined schemas for scientific and biomedical text.
"""

# Standard library
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # Fallback for Python < 3.8

# Local imports
from .catalog import PRESETS, get_schema, register_entity
from .data_handling.io import extract_entities

# Package version
try:
    # The package name is found in the [project] section of pyproject.toml
    __version__ = version("py-name-entity-recognition")
except PackageNotFoundError:
    # Fallback for when the package is not installed (e.g., in development)
    __version__ = "0.0.0-dev"

__all__ = [
    "extract_entities",
    "get_schema",
    "register_entity",
    "PRESETS",
]