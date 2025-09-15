import pytest
from pydantic import BaseModel, Field

from py_name_entity_recognition.prompting.prompt_manager import (
    PromptManager,
    ZeroShotStructured,
)


class SampleSchema(BaseModel):
    """A sample schema for testing."""

    person: list[str] = Field(description="The name of a person.")
    location: list[str] = Field(description="A physical location.")


@pytest.fixture
def prompt_manager():
    """Provides a PromptManager with the zero-shot strategy."""
    return PromptManager(strategy=ZeroShotStructured())


def test_zero_shot_prompt_generation(prompt_manager):
    template = prompt_manager.get_prompt_template(schema=SampleSchema)

    # Check that the template has the correct input variable
    assert "text_input" in template.input_variables

    # Check that the system message contains the schema descriptions
    system_message = template.messages[0].prompt.template
    assert (
        "You must extract entities that match the following schema:" in system_message
    )
    assert "Person" in system_message
    assert "The name of a person." in system_message
    assert "Location" in system_message
    assert "A physical location." in system_message

    # Check that the human message has the placeholder
    human_message = template.messages[1].prompt.template
    assert "{text_input}" in human_message
