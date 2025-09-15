import json

import pytest
from pydantic import BaseModel, Field

from py_name_entity_recognition.core.engine import CoreEngine

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_text():
    return "Alice lives in Paris and Bob lives in London."


async def test_engine_lcel_path_success(fake_llm_factory, test_schema, sample_text):
    """Tests the end-to-end LCEL path with a successful extraction."""
    responses = [
        json.dumps({"Person": ["Alice", "Bob"], "Location": ["Paris", "London"]})
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="lcel")

    result_dict = dict(result)
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"
    assert result_dict["Paris"] == "S-Location"
    assert result_dict["London"] == "S-Location"
    assert result_dict["lives"] == "O"


async def test_engine_agentic_path_success(fake_llm_factory, test_schema, sample_text):
    """Tests the agentic path where validation succeeds on the first try."""
    responses = [
        json.dumps({"Person": ["Alice", "Bob"], "Location": ["Paris", "London"]})
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = dict(result)
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"


async def test_engine_agentic_path_refinement(
    fake_llm_factory, test_schema, sample_text
):
    """Tests the agentic path with one round of refinement."""
    responses = [
        json.dumps({"Person": ["Alice"], "Location": ["Zurich"]}),  # Bad
        json.dumps(
            {"Person": ["Alice", "Bob"], "Location": ["Paris", "London"]}
        ),  # Good
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=1)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = dict(result)
    assert result_dict["London"] == "S-Location"
    assert "Zurich" not in result_dict


async def test_engine_agentic_path_max_retries(
    fake_llm_factory, test_schema, sample_text
):
    """Tests that the agentic path gives up after max retries."""
    bad_response = json.dumps({"Person": [], "Location": ["Zurich"]})
    responses = [bad_response] * 4  # Initial call + 3 retries
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=3)

    result = await engine.run(sample_text, mode="agentic")

    # Should fail gracefully and return all 'O' tags
    for _, tag in result:
        assert tag == "O"


async def test_engine_agentic_self_correction(
    fake_llm_factory, test_schema, sample_text
):
    """Tests that the agentic mode self-corrects a hallucinated entity."""
    responses = [
        json.dumps(
            {"Person": ["Alice", "Zurich"], "Location": ["Paris"]}
        ),  # Hallucinated
        json.dumps({"Person": ["Alice"], "Location": ["Paris"]}),  # Corrected
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=1)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = dict(result)
    assert "Zurich" not in result_dict
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Paris"] == "S-Location"


async def test_engine_chunking_and_merging(fake_llm_factory, test_schema):
    """Tests the chunking and merging logic for long documents."""
    long_text = "Alice is a doctor in New York. " * 50

    responses = [
        json.dumps({"Person": ["Alice"], "Location": ["New York", "Old York"]}),
        json.dumps({"Person": ["Alice"], "Location": ["New York"]}),
    ]
    llm = fake_llm_factory(responses * 10)
    engine = CoreEngine(model=llm, schema=test_schema, chunk_size=100, chunk_overlap=20)

    result = await engine.run(long_text, mode="lcel")

    found_alice = any(token == "Alice" and tag == "S-Person" for token, tag in result)
    found_ny = any(token == "New" and tag == "B-Location" for token, tag in result)
    found_old_york = any("Old" in token for token, tag in result)

    assert found_alice
    assert found_ny
    assert not found_old_york


async def test_engine_merging_overlapping_entities(fake_llm_factory, test_schema):
    """Tests that the merger correctly de-duplicates entities found in overlapping chunks."""
    text = (
        "Part one of the text is here. In the middle is Dr. Alice Smith. "
        "The final part of the text is over here."
    )
    # This text is set up so "Dr. Alice Smith" is fully contained in the overlap
    # of multiple chunks.

    # Mock LLM to return the entity for the chunks that contain it
    responses = [
        json.dumps({"Person": [], "Location": []}),
        json.dumps({"Person": ["Dr. Alice Smith"], "Location": []}),
        json.dumps({"Person": ["Dr. Alice Smith"], "Location": []}),
    ]
    llm = fake_llm_factory(responses)
    # Use chunk settings that cause "Dr. Alice Smith" to be in multiple chunks
    engine = CoreEngine(model=llm, schema=test_schema, chunk_size=60, chunk_overlap=40)

    result = await engine.run(text, mode="lcel")

    result_dict = dict(result)

    # Check that "Dr. Alice Smith" is correctly identified as one entity
    assert result_dict.get("Dr.") == "B-Person"
    assert result_dict.get("Alice") == "I-Person"
    assert result_dict.get("Smith") == "E-Person"

    # Count the number of "B-" tags to ensure the entity was not duplicated
    b_tag_count = sum(1 for _, tag in result if tag.startswith("B-"))
    assert b_tag_count == 1


async def test_engine_lcel_path_malformed_json(
    fake_llm_factory, test_schema, sample_text
):
    """Tests that the LCEL path raises a validation error for malformed JSON."""
    responses = ['{"Person": ["Alice"], "Location": "Paris"}']  # Malformed
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    # Pydantic v2 raises a ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await engine.run(sample_text, mode="lcel")


async def test_engine_no_entities_found(fake_llm_factory, test_schema, sample_text):
    """Tests the engine's behavior when no entities are found."""
    responses = [json.dumps({"Person": [], "Location": []})]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="lcel")

    for _, tag in result:
        assert tag == "O"


async def test_engine_agentic_path_case_sensitive_validation(
    fake_llm_factory, test_schema, sample_text
):
    """Tests that the validation is case-sensitive."""
    # LLM returns "london" (lowercase) which is not in the original text
    responses = [
        json.dumps({"Person": ["Alice"], "Location": ["london"]}),  # Bad casing
        json.dumps({"Person": ["Alice"], "Location": ["London"]}),  # Corrected
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema, max_retries=1)

    result = await engine.run(sample_text, mode="agentic")

    result_dict = dict(result)
    assert "london" not in result_dict
    assert result_dict["London"] == "S-Location"


async def test_engine_empty_input(fake_llm_factory, test_schema):
    """Tests that the engine handles empty string input gracefully."""
    llm = fake_llm_factory([])
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run("", mode="lcel")
    assert result == []


async def test_engine_whitespace_input(fake_llm_factory, test_schema):
    """Tests that the engine handles whitespace-only input gracefully."""
    llm = fake_llm_factory([])
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run("   \t\n  ", mode="lcel")
    assert result == []


async def test_engine_handles_none_in_entity_list(
    fake_llm_factory, test_schema, sample_text
):
    """Tests that the engine handles None values in the entity list gracefully."""
    # The Pydantic model might allow for List[Optional[str]]
    responses = [
        json.dumps({"Person": ["Alice", None, "Bob"], "Location": ["Paris", "London"]})
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=test_schema)

    result = await engine.run(sample_text, mode="lcel")

    # It should process the valid entities and ignore the None
    result_dict = dict(result)
    assert result_dict["Alice"] == "S-Person"
    assert result_dict["Bob"] == "S-Person"
    assert "None" not in result_dict


async def test_engine_with_complex_nested_schema(fake_llm_factory):
    """Tests the engine's ability to handle a more complex, nested Pydantic schema."""

    class FlightDetails(BaseModel):
        flight_number: str = Field(description="The flight number.")
        airline: str = Field(description="The airline carrier.")

    class Trip(BaseModel):
        traveler: str = Field(description="Name of the person traveling.")
        destination: str = Field(description="The final destination city.")
        flight: FlightDetails = Field(description="Details about the flight.")

    class Booking(BaseModel):
        """Schema for extracting travel booking details."""

        trip_details: Trip

    text = "Mr. Smith is booked on flight BA2490 to London with British Airways."
    llm_response = {
        "trip_details": {
            "traveler": "Mr. Smith",
            "destination": "London",
            "flight": {"flight_number": "BA2490", "airline": "British Airways"},
        }
    }
    responses = [json.dumps(llm_response)]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=Booking)

    result = await engine.run(text, mode="lcel")

    # The CoreEngine flattens the structure, so we check for the leaf values.
    # The entity type is the capitalized version of the Pydantic field name.
    result_dict = dict(result)

    assert result_dict["Mr."] == "B-Traveler"
    assert result_dict["Smith"] == "E-Traveler"
    assert result_dict["London"] == "S-Destination"
    assert result_dict["BA2490"] == "S-Flight_number"
    assert result_dict["British"] == "B-Airline"
    assert result_dict["Airways"] == "E-Airline"
