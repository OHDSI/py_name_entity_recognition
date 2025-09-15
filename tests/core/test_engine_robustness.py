import json

import pytest
from pydantic import BaseModel, Field

from py_name_entity_recognition.core.engine import CoreEngine

pytestmark = pytest.mark.asyncio


class RobustnessTestSchema(BaseModel):
    """A schema for robustness testing."""

    Person: list[str] = Field(description="The name of a person.")
    Location: list[str] = Field(description="The name of a city, state, or country.")
    Organization: list[str] = Field(
        description="The name of a company or organization."
    )


@pytest.mark.xfail(
    reason="Merger cannot currently re-construct entities split across chunks."
)
async def test_engine_merging_split_entity(fake_llm_factory):
    """
    Tests if the engine can merge an entity split across chunk boundaries.
    For example, "San" is in chunk 1 and "Francisco" is in chunk 2.
    This is marked as xfail as it's a known architectural limitation.
    """
    text = "The journey started in San Francisco and ended in London."
    # Chunk settings to split "Francisco"
    # Chunk 1: "The journey started in San "
    # Chunk 2: "Francisco and ended in London."

    # Mock LLM to return partial matches from each chunk
    responses = [
        json.dumps({"Person": [], "Location": [], "Organization": []}),  # From chunk 1
        json.dumps(
            {"Person": [], "Location": ["Francisco", "London"], "Organization": []}
        ),  # From chunk 2
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(
        model=llm,
        schema=RobustnessTestSchema,
        chunk_size=28,
        chunk_overlap=5,
    )

    result = await engine.run(text, mode="lcel")
    result_str = " ".join([f"{token}/{tag}" for token, tag in result])

    # The ideal behavior is that "San Francisco" is merged.
    assert "San/B-Location Francisco/E-Location" in result_str


async def test_engine_merging_conflicting_entity_types(fake_llm_factory):
    """
    Tests how the merger handles conflicting entity types for the same text
    from different overlapping chunks. The merger should prioritize the entity
    with the higher confidence score.
    """
    text = "He works at the University of New York."

    # Chunk 1: "He works at the University of New" -> "University of New" is an ORG
    # Chunk 2: "at the University of New York." -> "New York" is a GPE
    # "University of New" is more central and should win.
    responses = [
        json.dumps(
            {"Person": [], "Location": [], "Organization": ["University of New"]}
        ),
        json.dumps({"Person": [], "Location": ["New York"], "Organization": []}),
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(
        model=llm,
        schema=RobustnessTestSchema,
        chunk_size=35,
        chunk_overlap=20,
    )

    result = await engine.run(text, mode="lcel")
    result_str = " ".join([f"{token}/{tag}" for token, tag in result])

    # The higher-confidence entity (ORG) should be kept.
    assert (
        "University/B-Organization of/I-Organization New/E-Organization" in result_str
    )
    # The lower-confidence, overlapping entity (GPE) should be discarded.
    assert "New/B-Location" not in result_str
    assert "York/S-Location" not in result_str


async def test_engine_nested_entities(fake_llm_factory):
    """
    Tests how the engine handles nested entities. The merger should prioritize
    the longer entity span.
    """
    text = "The University of Washington is a great school in Washington."

    # Mock LLM to return both the nested and the larger entity.
    responses = [
        json.dumps(
            {
                "Person": [],
                "Organization": ["University of Washington"],
                "Location": ["Washington"],
            }
        )
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=RobustnessTestSchema)

    result = await engine.run(text, mode="lcel")
    result_str = " ".join([f"{token}/{tag}" for token, tag in result])

    # Check that the larger entity "University of Washington" is identified correctly.
    assert (
        "University/B-Organization of/I-Organization Washington/E-Organization"
        in result_str
    )

    # Check that the second, standalone "Washington" is identified as a location.
    # Note: The result_str will contain two "Washington" tokens.
    assert "in/O Washington/S-Location ./O" in result_str

    # Explicitly check that the first "Washington" is NOT tagged as a location,
    # because it's part of the longer ORG entity.
    assert "of/I-Organization Washington/S-Location" not in result_str


async def test_engine_with_unicode_and_special_chars(fake_llm_factory):
    """
    Tests that the engine handles unicode characters (like emojis and non-ASCII
    letters) and other special characters gracefully.
    """
    text = "Zoe Smith (zoe@example.com) from Montréal loves 🍕 and works at αβγ Inc."

    class UnicodeSchema(BaseModel):
        """Schema for unicode test."""

        Person: list[str] = Field(description="Name of the person.")
        Location: list[str] = Field(description="Name of the location.")
        Organization: list[str] = Field(description="Name of the organization.")

    responses = [
        json.dumps(
            {
                "Person": ["Zoe Smith"],
                "Location": ["Montréal"],
                "Organization": ["αβγ Inc."],
            }
        )
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=UnicodeSchema)

    result = await engine.run(text, mode="lcel")
    result_dict = dict(result)

    # Check that entities with special characters are correctly identified.
    assert result_dict.get("Zoe") == "B-Person"
    assert result_dict.get("Smith") == "E-Person"
    assert result_dict.get("Montréal") == "S-Location"
    assert result_dict.get("αβγ") == "B-Organization"
    assert result_dict.get("Inc.") == "E-Organization"

    # Check that other tokens are handled correctly.
    assert result_dict.get("🍕") == "O"
    assert result_dict.get("zoe@example.com") == "O"


async def test_engine_with_list_of_entities(fake_llm_factory):
    """
    Tests that the engine can extract all entities from a list.
    """
    text = "The team includes Alice, Bob, and Carol."

    responses = [
        json.dumps(
            {"Person": ["Alice", "Bob", "Carol"], "Location": [], "Organization": []}
        )
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=RobustnessTestSchema)

    result = await engine.run(text, mode="lcel")
    result_dict = dict(result)

    assert result_dict.get("Alice") == "S-Person"
    assert result_dict.get("Bob") == "S-Person"
    assert result_dict.get("Carol") == "S-Person"


async def test_engine_ambiguous_entities(fake_llm_factory):
    """
    Tests that the engine can disambiguate entities based on context.
    "Washington" can be a person, location, or organization.
    """
    text = "George Washington visited Washington, D.C., not the state of Washington."

    responses = [
        json.dumps(
            {
                "Person": ["George Washington"],
                "Location": ["Washington, D.C.", "Washington"],
                "Organization": [],
            }
        )
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=RobustnessTestSchema)

    result = await engine.run(text, mode="lcel")
    result_str = " ".join([f"{token}/{tag}" for token, tag in result])

    # Check that "George Washington" is a person
    assert "George/B-Person Washington/E-Person" in result_str
    # Check that "Washington, D.C." is a location
    assert "Washington/B-Location ,/I-Location D.C./E-Location" in result_str
    # Check that the standalone "Washington" is a location
    assert "of/O Washington/S-Location ./O" in result_str


async def test_engine_with_no_entities_in_text(fake_llm_factory):
    """
    Tests that the engine returns an empty result when no entities are found.
    """
    text = "This is a sentence with no people, places, or companies."

    # Mock LLM to return an empty JSON object
    responses = [json.dumps({"Person": [], "Location": [], "Organization": []})]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=RobustnessTestSchema)

    result = await engine.run(text, mode="lcel")

    # The result should be a list of tuples with all 'O' tags
    all_o_tags = all(tag == "O" for _, tag in result)
    assert all_o_tags
    # Verify the length is correct by tokenizing the original text
    # using the engine's tokenizer
    doc = engine.biores_converter.nlp(text)
    assert len(result) == len([token.text for token in doc])
