"""
pyNameEntityRecognition: A Python package for state-of-the-art LLM-based Named Entity Recognition.

This package leverages LangChain for LLM orchestration and LangGraph for creating robust,
agentic, and self-refining extraction workflows. It includes a comprehensive catalog
of predefined schemas for scientific and biomedical text.
"""

__version__ = "1.0.6"

# Expose the primary user-facing function for easy access.
# Expose the catalog features for schema customization and extension.
from .catalog import PRESETS, get_schema, register_entity
from .data_handling.io import extract_entities

__all__ = [
    "extract_entities",
    "get_schema",
    "register_entity",
    "PRESETS",
]
