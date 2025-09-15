from typing import Optional

import pytest
from pydantic import BaseModel

from py_name_entity_recognition.core.engine import CoreEngine
from py_name_entity_recognition.schemas.core_schemas import BaseEntity


class SimpleSchema(BaseModel):
    Person: list[str]
    Location: str


class NestedSchema(BaseModel):
    class FlightDetails(BaseModel):
        flight_number: str
        airline: str

    traveler: str
    flight: FlightDetails


class ListOfModelsSchema(BaseModel):
    class Passenger(BaseModel):
        name: str
        seat: str

    passengers: list[Passenger]


class DeeplyNestedSchema(BaseModel):
    class Level2(BaseModel):
        class Level3(BaseModel):
            final_value: str

        level3_data: Level3

    level2_data: Level2


@pytest.fixture
def engine(fake_llm_factory):
    """
    Fixture to provide a CoreEngine instance.
    A dummy model and schema are used for instantiation, as they are not
    relevant for the method being tested.
    """
    llm = fake_llm_factory([])
    return CoreEngine(model=llm, schema=BaseModel)


def test_transform_simple_schema(engine):
    """Tests flattening a simple Pydantic model with a list and a string."""
    model = SimpleSchema(Person=["Alice", "Bob"], Location="Paris")
    result = engine._transform_to_base_entities(model)

    assert len(result) == 3
    # Use a set for easier comparison, as order doesn't matter.
    result_set = set(result)
    expected_set = {
        BaseEntity(type="Person", text="Alice"),
        BaseEntity(type="Person", text="Bob"),
        BaseEntity(type="Location", text="Paris"),
    }
    assert result_set == expected_set


def test_transform_nested_schema(engine):
    """Tests flattening a model with a nested Pydantic model."""
    model = NestedSchema(
        traveler="John Doe",
        flight=NestedSchema.FlightDetails(
            flight_number="BA2490", airline="British Airways"
        ),
    )
    result = engine._transform_to_base_entities(model)

    assert len(result) == 3
    result_set = set(result)
    expected_set = {
        BaseEntity(type="Traveler", text="John Doe"),
        BaseEntity(type="Flight_number", text="BA2490"),
        BaseEntity(type="Airline", text="British Airways"),
    }
    assert result_set == expected_set


def test_transform_list_of_models_schema(engine):
    """Tests flattening a model with a list of nested Pydantic models."""
    model = ListOfModelsSchema(
        passengers=[
            ListOfModelsSchema.Passenger(name="Alice", seat="1A"),
            ListOfModelsSchema.Passenger(name="Bob", seat="2B"),
        ]
    )
    result = engine._transform_to_base_entities(model)

    assert len(result) == 4
    result_set = set(result)
    expected_set = {
        BaseEntity(type="Name", text="Alice"),
        BaseEntity(type="Seat", text="1A"),
        BaseEntity(type="Name", text="Bob"),
        BaseEntity(type="Seat", text="2B"),
    }
    assert result_set == expected_set


def test_transform_deeply_nested_schema(engine):
    """Tests flattening a model with multiple levels of nesting."""
    model = DeeplyNestedSchema(
        level2_data=DeeplyNestedSchema.Level2(
            level3_data=DeeplyNestedSchema.Level2.Level3(final_value="Success")
        )
    )
    result = engine._transform_to_base_entities(model)
    assert len(result) == 1
    assert result[0] == BaseEntity(type="Final_value", text="Success")


def test_transform_empty_model(engine):
    """Tests that an empty Pydantic model results in an empty list."""

    class EmptySchema(BaseModel):
        pass

    model = EmptySchema()
    result = engine._transform_to_base_entities(model)
    assert result == []


def test_transform_model_with_none_values(engine):
    """Tests that None values in the model are ignored during transformation."""

    class SchemaWithOptional(BaseModel):
        name: str
        location: Optional[str] = None
        friends: list[Optional[str]]

    model = SchemaWithOptional(
        name="Alice", location=None, friends=["Bob", None, "Charlie"]
    )
    result = engine._transform_to_base_entities(model)

    assert len(result) == 3
    result_set = set(result)
    expected_set = {
        BaseEntity(type="Name", text="Alice"),
        BaseEntity(type="Friends", text="Bob"),
        BaseEntity(type="Friends", text="Charlie"),
    }
    assert result_set == expected_set
    # Double-check that no entity was created for the None values
    assert not any(entity.text is None for entity in result)
    assert not any(entity.type == "Location" for entity in result)


def test_transform_with_empty_list(engine):
    """Tests a model where one of the fields is an empty list."""
    model = SimpleSchema(Person=[], Location="Paris")
    result = engine._transform_to_base_entities(model)

    assert len(result) == 1
    assert result[0] == BaseEntity(type="Location", text="Paris")


def test_transform_with_empty_string(engine):
    """Tests that empty strings are ignored during transformation."""
    model = SimpleSchema(Person=["Alice", ""], Location="Paris")
    result = engine._transform_to_base_entities(model)

    assert len(result) == 2
    result_set = set(result)
    expected_set = {
        BaseEntity(type="Person", text="Alice"),
        BaseEntity(type="Location", text="Paris"),
    }
    assert result_set == expected_set
