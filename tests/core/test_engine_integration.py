import os

import pytest
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from py_name_entity_recognition.core.engine import CoreEngine

# Marker for integration tests
pytestmark = pytest.mark.integration

# Skip all tests in this module if OpenAI API key is not present
if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY not found, skipping integration tests", allow_module_level=True
    )


# Schema for testing
class IntegrationTestSchema(BaseModel):
    """A simple schema for integration testing."""

    Person: list[str] = Field(description="The name of a person.")
    City: list[str] = Field(description="The name of a city.")


# Fixture to select LLM based on environment variables
@pytest.fixture(scope="module")
def llm():
    if os.environ.get("OPENAI_API_KEY"):
        # Ensure you have set the OPENAI_API_KEY environment variable
        return ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        # Fallback to a local Ollama instance
        # Ensure Ollama is running and the model (e.g., 'llama3') is pulled
        try:
            llm = Ollama(model="llama3")
            # A quick check to see if the server is responsive
            llm.invoke("Hi")
            return llm
        except Exception:
            pytest.skip(
                "Ollama server not found or model not available. Skipping integration tests."
            )


@pytest.mark.asyncio
async def test_engine_lcel_mode_integration(llm):
    """
    Integration test for the LCEL mode with a real LLM.
    This test calls an external LLM and may incur costs.
    """
    text = "John Doe is a software engineer who lives in San Francisco."
    engine = CoreEngine(model=llm, schema=IntegrationTestSchema)

    result = await engine.run(text, mode="lcel")

    result_dict = dict(result)

    # Assert that the core entities are found.
    # Note: Real LLM output can be non-deterministic, so assertions are kept simple.
    assert result_dict.get("John") == "B-Person"
    assert result_dict.get("Doe") == "E-Person"
    assert "San" in result_dict
    assert "Francisco" in result_dict

    # Check that "San Francisco" is tagged as a city
    sf_tags = [result_dict.get("San"), result_dict.get("Francisco")]
    assert "B-City" in sf_tags
    assert "E-City" in sf_tags


@pytest.mark.asyncio
async def test_engine_agentic_mode_hallucination_correction(llm):
    """
    Integration test for the agentic mode's ability to correct hallucinations.
    """
    # This text does not contain a 'JobTitle'
    text = "Jane Doe is a biologist from Berlin."

    class HallucinationTestSchema(BaseModel):
        """A schema designed to potentially induce a hallucination."""

        Person: list[str] = Field(description="The name of a person.")
        JobTitle: list[str] = Field(
            description="The job title of the person. If not present, do not extract."
        )

    # The agentic mode should be smart enough to see "JobTitle" is not in the text
    # and not extract "biologist".
    engine = CoreEngine(model=llm, schema=HallucinationTestSchema, max_retries=2)

    result = await engine.run(text, mode="agentic")

    result_dict = dict(result)

    # Ensure the person is correctly identified
    assert result_dict.get("Jane") == "B-Person"
    assert result_dict.get("Doe") == "E-Person"

    # Crucially, check that "biologist" was NOT extracted as a JobTitle
    assert "biologist" not in result_dict or result_dict["biologist"] == "O"
