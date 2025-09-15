from py_name_entity_recognition.data_handling.io import biores_to_entities
from py_name_entity_recognition.schemas.core_schemas import BaseEntity, Entities


def test_biores_to_entities_simple():
    """Tests conversion of simple B-I-O-E-S tags."""
    tagged_tokens = [
        ("John", "B-Person"),
        ("Doe", "E-Person"),
        ("lives", "O"),
        ("in", "O"),
        ("New", "B-Location"),
        ("York", "E-Location"),
        (".", "O"),
        ("Microsoft", "S-Company"),
    ]
    expected_entities = Entities(
        entities=[
            BaseEntity(type="Person", text="John Doe"),
            BaseEntity(type="Location", text="New York"),
            BaseEntity(type="Company", text="Microsoft"),
        ]
    )
    result = biores_to_entities(tagged_tokens)
    assert result == expected_entities


def test_biores_to_entities_with_i_tags():
    """Tests conversion with intermediate (I) tags."""
    tagged_tokens = [
        ("Dr.", "B-Person"),
        ("Emily", "I-Person"),
        ("Carter", "E-Person"),
    ]
    expected_entities = Entities(
        entities=[BaseEntity(type="Person", text="Dr. Emily Carter")]
    )
    result = biores_to_entities(tagged_tokens)
    assert result == expected_entities


def test_biores_to_entities_no_entities():
    """Tests conversion when there are no entities."""
    tagged_tokens = [("some", "O"), ("random", "O"), ("text", "O")]
    expected_entities = Entities(entities=[])
    result = biores_to_entities(tagged_tokens)
    assert result == expected_entities


def test_biores_to_entities_empty_input():
    """Tests conversion with empty input."""
    tagged_tokens = []
    expected_entities = Entities(entities=[])
    result = biores_to_entities(tagged_tokens)
    assert result == expected_entities


def test_biores_to_entities_ends_with_entity():
    """Tests conversion when the text ends with an entity."""
    tagged_tokens = [("call", "O"), ("me", "O"), ("Ishmael", "S-Person")]
    expected_entities = Entities(entities=[BaseEntity(type="Person", text="Ishmael")])
    result = biores_to_entities(tagged_tokens)
    assert result == expected_entities
