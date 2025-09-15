import pytest
from pydantic import BaseModel

from py_name_entity_recognition import catalog


def test_get_schema_with_preset():
    """Test that a valid preset returns a schema with the correct entities."""
    schema = catalog.get_schema(preset="CLINICAL_TRIAL_CORE")
    assert issubclass(schema, BaseModel)
    assert len(schema.model_fields) == len(catalog.PRESETS["CLINICAL_TRIAL_CORE"])
    assert "DiseaseOrSyndrome" in schema.model_fields
    assert "AdverseEvent" in schema.model_fields


def test_get_schema_with_invalid_preset_raises_value_error():
    """Test that an invalid preset raises a ValueError."""
    with pytest.raises(ValueError, match="Preset 'INVALID_PRESET' not found"):
        catalog.get_schema(preset="INVALID_PRESET")


def test_get_schema_with_include_categories():
    """Test that including categories returns a schema with the correct entities."""
    categories = ["CHEMICALS_AND_DRUGS"]
    schema = catalog.get_schema(include_categories=categories)
    assert issubclass(schema, BaseModel)
    for entity, definition in catalog.ENTITY_REGISTRY.items():
        if definition["category"] in categories:
            assert entity in schema.model_fields
        else:
            assert entity not in schema.model_fields


def test_get_schema_with_include_entities():
    """Test that including specific entities returns a schema with those entities."""
    entities = ["GeneOrGenome", "Protein"]
    schema = catalog.get_schema(include_entities=entities)
    assert issubclass(schema, BaseModel)
    assert len(schema.model_fields) == len(entities)
    assert "GeneOrGenome" in schema.model_fields
    assert "Protein" in schema.model_fields


def test_get_schema_with_exclude_entities():
    """Test that excluding entities removes them from the schema."""
    preset = "MOLECULAR_BIOLOGY"
    exclude = ["GeneOrGenome", "Protein"]
    schema = catalog.get_schema(preset=preset, exclude_entities=exclude)
    assert issubclass(schema, BaseModel)
    assert "GeneOrGenome" not in schema.model_fields
    assert "Protein" not in schema.model_fields
    assert "CellType" in schema.model_fields


def test_get_schema_with_category_and_exclude():
    """Test combining category inclusion with entity exclusion."""
    categories = ["CHEMICALS_AND_DRUGS", "DISORDERS_AND_FINDINGS"]
    exclude = ["PharmacologicSubstance"]  # Exclude one entity from the category

    schema = catalog.get_schema(include_categories=categories, exclude_entities=exclude)

    assert issubclass(schema, BaseModel)
    # Check that an entity from the excluded list is NOT present
    assert "PharmacologicSubstance" not in schema.model_fields
    # Check that other entities from the included categories ARE present
    assert "DiseaseOrSyndrome" in schema.model_fields
    assert "SignOrSymptom" in schema.model_fields


def test_get_schema_exclude_nonexistent_entity_is_ignored():
    """Test that trying to exclude an entity that doesn't exist is ignored."""
    preset = "CLINICAL_TRIAL_CORE"
    original_schema = catalog.get_schema(preset=preset)

    # "NonExistentEntity" is not in the preset or the registry
    exclude = ["NonExistentEntity"]
    schema = catalog.get_schema(preset=preset, exclude_entities=exclude)

    # The schema should be identical to the one without the invalid exclusion
    assert len(schema.model_fields) == len(original_schema.model_fields)


def test_get_schema_include_nonexistent_entity_raises_error():
    """Test that trying to include an entity that doesn't exist raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="The following entities from 'include_entities' are not in the registry: NonExistentEntity",
    ):
        catalog.get_schema(include_entities=["GeneOrGenome", "NonExistentEntity"])


def test_get_schema_with_empty_result_raises_value_error():
    """Test that a combination resulting in no entities raises a ValueError."""
    with pytest.raises(ValueError, match="resulted in an empty set of entities"):
        catalog.get_schema(
            include_entities=["GeneOrGenome"], exclude_entities=["GeneOrGenome"]
        )


def test_get_schema_defaults_to_comprehensive():
    """Test that calling get_schema without arguments returns the comprehensive set."""
    schema = catalog.get_schema()
    assert issubclass(schema, BaseModel)
    assert len(schema.model_fields) == len(catalog.ENTITY_REGISTRY)


def test_register_entity():
    """Test that a new entity can be registered."""
    key = "CustomBiomarker"
    definition = {
        "name": "Custom Biomarker",
        "description": "A test biomarker.",
        "category": "TESTING",
    }
    catalog.register_entity(key, definition)
    assert key in catalog.ENTITY_REGISTRY
    assert catalog.ENTITY_REGISTRY[key] == definition


def test_register_entity_with_existing_key_raises_value_error():
    """Test that registering an existing entity without overwrite raises a ValueError."""
    key = "DiseaseOrSyndrome"
    definition = {
        "name": "Test",
        "description": "Test",
        "category": "TESTING",
    }
    with pytest.raises(ValueError, match=f"Entity key '{key}' already exists"):
        catalog.register_entity(key, definition)


def test_register_entity_with_overwrite():
    """Test that an existing entity can be overwritten."""
    key = "DiseaseOrSyndrome"
    original_definition = catalog.ENTITY_REGISTRY[key]
    new_definition = {
        "name": "Overwritten Disease",
        "description": "An overwritten description.",
        "category": "OVERWRITTEN",
    }
    catalog.register_entity(key, new_definition, overwrite=True)
    assert catalog.ENTITY_REGISTRY[key] == new_definition
    # Restore the original definition to not affect other tests
    catalog.ENTITY_REGISTRY[key] = original_definition
