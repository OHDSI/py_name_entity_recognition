from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    """
    Represents a single extracted entity span.

    This is the intermediate format that the LLM is expected to produce.
    """

    class Config:
        frozen = True

    type: str = Field(
        ...,
        description="The category or type of the extracted entity (e.g., 'Disease', 'Symptom').",
    )
    text: str = Field(
        ...,
        description="The actual text span that was extracted from the source document.",
    )


class Entities(BaseModel):
    """
    A container for a list of extracted entities.

    The LLM is expected to return a JSON object that conforms to this schema.
    """

    entities: list[BaseEntity] = Field(
        ..., description="A list of all entities extracted from the text."
    )
