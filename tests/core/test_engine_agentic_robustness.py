import json

import pytest
from pydantic import BaseModel, Field

from py_name_entity_recognition.core.engine import CoreEngine

pytestmark = pytest.mark.asyncio


class AgenticTestSchema(BaseModel):
    """A schema for agentic mode robustness testing."""

    Person: list[str] = Field(description="The name of a person.")
    Organization: list[str] = Field(
        description="The name of a company or organization."
    )


async def test_agentic_mode_anti_hallucination(fake_llm_factory):
    """
    Tests that the agentic mode's validation step correctly identifies and
    discards hallucinated entities that are not verbatim matches in the source text.
    """
    text = "Dr. Eleanor Vance is a researcher at the Institute of Science."

    # Mock LLM to return a hallucinated entity "CyberCorp" and a real one.
    # The agentic validator should discard "CyberCorp".
    responses = [
        json.dumps(
            {
                "Person": ["Dr. Eleanor Vance"],
                "Organization": ["Institute of Science", "CyberCorp"],
            }
        )
    ]
    llm = fake_llm_factory(responses)
    engine = CoreEngine(model=llm, schema=AgenticTestSchema)

    # Run the engine in 'agentic' mode
    result = await engine.run(text, mode="agentic")
    result_str = " ".join([f"{token}/{tag}" for token, tag in result])

    # Check that the valid entity "Institute of Science" is present
    assert (
        "Institute/B-Organization of/I-Organization Science/E-Organization"
        in result_str
    )

    # Check that the hallucinated entity "CyberCorp" is NOT present
    assert "CyberCorp" not in result_str
    # Check that no token is tagged as an organization other than the valid one
    org_tags = [tag for _, tag in result if "Organization" in tag]
    assert len(org_tags) == 3  # B-Organization, I-Organization, E-Organization
