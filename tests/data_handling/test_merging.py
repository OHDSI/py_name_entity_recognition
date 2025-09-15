import re

import pytest

from py_name_entity_recognition.data_handling.merging import ChunkMerger
from py_name_entity_recognition.schemas.core_schemas import BaseEntity


@pytest.fixture
def merger():
    return ChunkMerger()


def test_calculate_confidence_scores(merger):
    """Tests the confidence calculation logic."""
    # Entity exactly at the center of the chunk
    # Chunk: [0, 100), Entity starts at 50
    assert merger._calculate_confidence(50, 0, 100) == 1.0

    # Entity at the start of the chunk
    # Chunk: [0, 100), Entity starts at 0
    assert merger._calculate_confidence(0, 0, 100) == 0.0

    # Entity at the end of the chunk
    # Chunk: [50, 150), Entity starts at 140 (text length 10)
    assert merger._calculate_confidence(140, 50, 150) == pytest.approx(0.2)

    # Entity in the first quarter of the chunk
    # Chunk: [0, 100), Entity starts at 25
    assert merger._calculate_confidence(25, 0, 100) == 0.5

    # Zero-length chunk
    assert merger._calculate_confidence(0, 0, 0) == 0.0


def test_merge_no_overlap(merger):
    """Tests merging entities from two distinct, non-overlapping chunks."""
    full_text = "Alice lives in Paris and Bob lives in London."
    # Chunk 1: "Alice lives in Paris"
    entities1 = [BaseEntity(type="Person", text="Alice")]
    chunk1 = (entities1, 0, 20)
    # Chunk 2: "Bob lives in London."
    entities2 = [BaseEntity(type="Person", text="Bob")]
    chunk2 = (entities2, 25, len(full_text))

    result = merger.merge(full_text, [chunk1, chunk2])
    result_dict = dict(result)

    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"
    assert result_dict["Paris"] == "O"  # Not in any chunk's entity list
    assert result_dict["London"] == "O"


def test_merge_with_overlap_and_duplicate(merger):
    """Tests that an entity found in two overlapping chunks is merged correctly."""
    full_text = "The event is in New York City tomorrow."
    entity = BaseEntity(type="Location", text="New York City")

    # Chunk 1 contains the entity and is well-centered
    chunk1 = ([entity], 10, 35)  # "in New York City tomo"
    # Chunk 2 also contains the entity, but it's at the edge
    chunk2 = ([entity], 0, 25)  # "The event is in New York"

    result = merger.merge(full_text, [chunk1, chunk2])
    result_dict = dict(result)

    # "New York City" should be identified as a single entity
    assert result_dict["New"] == "B-Location"
    assert result_dict["York"] == "I-Location"
    assert result_dict["City"] == "E-Location"

    # Count "B-" tags to ensure no duplicates
    b_tag_count = sum(1 for _, tag in result if tag.startswith("B-"))
    assert b_tag_count == 1


def test_merge_with_overlap_and_conflict(merger):
    """
    Tests resolving a conflict where two chunks identify the same text
    as different entity types. The one with the higher confidence score wins.
    """
    full_text = "Talk about the new Apple iPhone today."
    # "Apple" is the text in conflict. It's at index 19.

    # Chunk 1: "new Apple iPhon" - "Apple" is centered, should have high confidence
    entities1 = [BaseEntity(type="Corporation", text="Apple")]
    chunk1 = (entities1, 15, 30)  # chunk text is full_text[15:30]

    # Chunk 2: "Talk about the new Apple" - "Apple" is at the edge, low confidence
    entities2 = [BaseEntity(type="Fruit", text="Apple")]
    chunk2 = (entities2, 0, 24)  # chunk text is full_text[0:24]

    result = merger.merge(full_text, [chunk1, chunk2])
    result_dict = dict(result)

    # "Apple" should be tagged as "Corporation", not "Fruit"
    assert result_dict["Apple"] == "S-Corporation"


def test_merge_multiple_occurrences_of_same_text(merger):
    """
    Tests that if the same entity text appears multiple times, the merger
    correctly identifies and tags all distinct occurrences.
    """
    full_text = "Paris is a city, and the other Paris is a person."
    # Chunk 1 covers the first "Paris"
    entities1 = [BaseEntity(type="City", text="Paris")]
    chunk1 = (entities1, 0, 20)
    # Chunk 2 covers the second "Paris"
    entities2 = [BaseEntity(type="Person", text="Paris")]
    chunk2 = (entities2, 21, len(full_text))

    result = merger.merge(full_text, [chunk1, chunk2])

    # The first "Paris" should be a City
    assert result[0] == ("Paris", "S-City")
    # The second "Paris" should be a Person. It's the 9th token (index 8).
    assert result[8] == ("Paris", "S-Person")


def test_merge_empty_chunk_results(merger):
    """Tests that the merger handles an empty list of chunk results."""
    full_text = "Some text with no entities."
    result = merger.merge(full_text, [])
    for _, tag in result:
        assert tag == "O"


def test_merge_chunks_with_no_entities(merger):
    """Tests merging chunks that contain no entities."""
    full_text = "Alice lives in Paris."
    chunk1 = ([], 0, 12)
    chunk2 = ([], 10, len(full_text))

    result = merger.merge(full_text, [chunk1, chunk2])
    for _, tag in result:
        assert tag == "O"


def test_merge_invalid_regex_in_entity(merger, monkeypatch):
    """
    Tests that an entity containing special regex characters that could form an
    invalid pattern is handled gracefully.
    """
    full_text = "He said (hello."
    # The entity text "(" is an invalid regex pattern
    entities = [BaseEntity(type="GREETING", text="(")]
    chunk_results = [(entities, 0, len(full_text))]

    # Mock re.finditer to simulate a regex error
    def mock_finditer(*args, **kwargs):
        raise re.error("test error")

    monkeypatch.setattr("re.finditer", mock_finditer)

    result = merger.merge(full_text, chunk_results)

    # The merger should skip the problematic entity and tag everything as 'O'.
    for _, tag in result:
        assert tag == "O"
