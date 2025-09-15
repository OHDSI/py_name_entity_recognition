import json
from typing import Any

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class TestSchema(BaseModel):
    """A Pydantic schema for use in tests."""

    Person: list[str] = Field(description="The name of a person.")
    Location: list[str] = Field(
        description="A physical location, like a city or country."
    )


class FakeStructuredLLM(FakeListLLM):
    """
    A fake LLM that simulates the behavior of with_structured_output for testing.
    It inherits from FakeListLLM to provide canned string responses and overrides
    the structured output method to parse that string into a Pydantic model.
    """

    def with_structured_output(self, schema: dict, **kwargs: Any) -> Runnable:
        """
        Overrides the base method to return a chain that simulates parsing the
        fake LLM's JSON string response into a dictionary.
        """
        # The schema is now a dict. The FakeListLLM returns a JSON string,
        # so we just need to parse it into a dict to simulate the real behavior.
        return self | RunnableLambda(json.loads)


@pytest.fixture(scope="session")
def test_schema() -> type[BaseModel]:
    """Provides the TestSchema class."""
    return TestSchema


@pytest.fixture
def fake_llm_factory():
    """
    A factory fixture to create FakeStructuredLLM instances.
    This allows tests to simulate specific LLM JSON outputs and test chains
    that rely on the with_structured_output method.
    """

    def _factory(responses: list[str]):
        return FakeStructuredLLM(responses=responses)

    return _factory
