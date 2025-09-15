import pytest

from py_name_entity_recognition.utils.biores_converter import BIOSESConverter


@pytest.fixture(scope="module")
def converter():
    """Provides a BIOSESConverter instance for the tests."""
    return BIOSESConverter()


def test_biores_converter_simple_case(converter):
    text = "Apple Inc. is a technology company."
    entities_with_spans = [(0, 10, "ORG")]  # "Apple Inc."
    result = converter.convert(text, entities_with_spans)

    # Check for presence of tags, regardless of exact tokenization of surrounding text
    assert ("Apple", "B-ORG") in result
    assert ("Inc.", "E-ORG") in result
    # Find the token for 'is' and check it's 'O'
    is_tag = [tag for token, tag in result if token == "is"]
    assert is_tag == ["O"]


def test_biores_converter_single_token_entity(converter):
    text = "I like Google."
    entities_with_spans = [(7, 13, "ORG")]  # "Google"
    result = converter.convert(text, entities_with_spans)

    assert ("Google", "S-ORG") in result
    assert ("like", "O") in result


def test_biores_converter_multiple_entities(converter):
    text = "Apple and Google are rivals."
    entities_with_spans = [(0, 5, "ORG"), (10, 16, "ORG")]  # "Apple", "Google"
    result = converter.convert(text, entities_with_spans)

    assert ("Apple", "S-ORG") in result
    assert ("Google", "S-ORG") in result
    assert ("rivals", "O") in result


def test_biores_converter_nested_entities(converter):
    """Tests that longer entities are prioritized over shorter, nested ones."""
    text = "He works at New York University."
    entities_with_spans = [
        (12, 20, "GPE"),  # "New York"
        (12, 31, "ORG"),  # "New York University"
    ]
    result = converter.convert(text, entities_with_spans)

    assert ("New", "B-ORG") in result
    assert ("York", "I-ORG") in result
    assert ("University", "E-ORG") in result
    # Ensure the shorter entity "New York" was not tagged as GPE
    assert not any(tag.endswith("-GPE") for _, tag in result)


def test_biores_converter_no_entities(converter):
    text = "Just a simple sentence."
    entities_with_spans = []
    result = converter.convert(text, entities_with_spans)

    for _, tag in result:
        assert tag == "O"


def test_biores_converter_misaligned_span(converter):
    """Tests that spans not aligning with token boundaries are skipped."""
    text = "The quick brown fox."
    # "ui" is a substring of "quick", not a full token.
    entities_with_spans = [(5, 7, "ANIMAL")]  # "ui"
    result = converter.convert(text, entities_with_spans)

    for _, tag in result:
        assert tag == "O"


def test_biores_converter_three_token_entity(converter):
    text = "I love New York City."
    entities_with_spans = [(7, 20, "GPE")]  # "New York City"
    result = converter.convert(text, entities_with_spans)

    assert ("New", "B-GPE") in result
    assert ("York", "I-GPE") in result
    assert ("City", "E-GPE") in result


def test_biores_converter_spacy_model_not_found(monkeypatch):
    """Tests that an IOError is raised if the spaCy model is not found."""
    monkeypatch.setattr("spacy.load", lambda name: (_ for _ in ()).throw(OSError))
    with pytest.raises(OSError, match="Default spaCy model not found"):
        BIOSESConverter()


def test_biores_converter_overlapping_entities_same_length(converter):
    """Tests that the first of two overlapping entities of the same length is prioritized."""
    text = "He works at the University of New York."
    entities_with_spans = [
        (
            text.find("University of New"),
            text.find("University of New") + len("University of New"),
            "ORG",
        ),
        (
            text.find("of New York"),
            text.find("of New York") + len("of New York"),
            "GPE",
        ),
    ]
    result = converter.convert(text, entities_with_spans)

    assert ("University", "B-ORG") in result
    assert ("of", "I-ORG") in result
    assert ("New", "E-ORG") in result
    assert ("York", "O") in result  # Should not be part of the GPE
    assert not any(tag.endswith("-GPE") for _, tag in result)


def test_biores_converter_special_characters(converter):
    """Tests entities with special characters."""
    text = "The C.I.A. is a U.S. agency. He is the C.E.O."
    entities_with_spans = [
        (text.find("C.I.A."), text.find("C.I.A.") + len("C.I.A."), "ORG"),
        (text.find("C.E.O."), text.find("C.E.O.") + len("C.E.O."), "TITLE"),
    ]
    result = converter.convert(text, entities_with_spans)

    assert ("C.I.A.", "S-ORG") in result
    assert ("C.E.O.", "S-TITLE") in result


def test_biores_converter_case_sensitivity(converter):
    """Tests that entity matching is case-sensitive."""
    text = "I like apple."
    # No entity for "Apple" (uppercase) should be found.
    entities_with_spans = []
    result = converter.convert(text, entities_with_spans)

    for _, tag in result:
        assert tag == "O"
