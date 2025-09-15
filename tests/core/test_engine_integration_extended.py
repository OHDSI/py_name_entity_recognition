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


geekwire_article_text = """
A pivotal antitrust ruling aimed at addressing Google’s search monopoly may have handed Microsoft its best competitive opening in many years — but it’s still a slim one, and it’s not clear if the Redmond company will even consider it worth the effort.

In the long-awaited decision Tuesday afternoon, U.S. District Judge Amit P. Mehta barred Google from entering into exclusive contracts that make its search engine the default on browsers and smartphones, such as the search giant’s longstanding deal involving Apple’s iPhone.

But under the ruling, Google can still spend billions to secure default slots, as long as partners remain free to preload or promote rivals. That leaves Microsoft to fight both Google’s scale and its checkbook.

So as an additional fix, Mehta also ordered Google to syndicate its organic search results and text ads to rivals for up to five years on commercially reasonable terms. For Microsoft, that could mean the ability to offer its users search results and ads backed by Google’s industry-leading index.

That could help Microsoft’s Bing search engine — and by extension its Copilot chatbot — compete more effectively against Google’s search dominance. The stakes are higher than ever given the foundational role of search in the emerging world of artificial intelligence.

But it’s not clear if Microsoft will try to take advantage of the remedies. Contacted by GeekWire on Wednesday, a spokesperson said the company had no comment on the ruling.
"""


class ArticleSchema(BaseModel):
    """Schema for extracting information from a news article."""

    Person: list[str] = Field(
        description="The name of a person mentioned in the article."
    )
    Organization: list[str] = Field(
        description="The name of a company or organization."
    )
    Location: list[str] = Field(
        description="A city, state, or country mentioned in the article."
    )


@pytest.mark.asyncio
async def test_engine_lcel_mode_with_chunking_integration(llm):
    """
    Integration test for the LCEL mode with a real LLM on a long article
    that requires chunking.
    """
    engine = CoreEngine(
        model=llm, schema=ArticleSchema, chunk_size=200, chunk_overlap=50
    )

    result = await engine.run(geekwire_article_text, mode="lcel")

    result_dict = dict(result)

    # People
    assert "Amit P. Mehta" in " ".join(result_dict.keys())

    # Organizations
    assert "Google" in result_dict
    assert "Microsoft" in result_dict
    assert "Apple" in result_dict
    assert "GeekWire" in result_dict

    # Locations
    # (No locations in this snippet)


@pytest.mark.asyncio
async def test_engine_agentic_mode_refinement_integration(llm):
    """
    Integration test for the agentic mode's ability to refine its output.
    This test uses a scenario where the LLM might initially extract a plausible
    but incorrect entity.
    """
    text = "Microsoft’s Bing search engine — and by extension its Copilot chatbot — compete more effectively against Google’s search dominance."

    class RefinementSchema(BaseModel):
        """A schema that might cause the LLM to be overly aggressive."""

        Product: list[str] = Field(description="The name of a software product.")
        Feature: list[str] = Field(description="A feature of a software product.")

    engine = CoreEngine(model=llm, schema=RefinementSchema, max_retries=1)

    result = await engine.run(text, mode="agentic")

    result_dict = dict(result)

    # We want to ensure that "search dominance" is not extracted as a feature.
    # The agentic mode should recognize that this is not a feature.
    assert "search dominance" not in " ".join(result_dict.keys())
    assert "Bing" in result_dict
    assert "Copilot" in result_dict


@pytest.mark.asyncio
async def test_engine_with_numerical_and_date_entities(llm):
    """
    Integration test for extracting numerical and date entities.
    """
    text = "The court noted that Google paid an estimated $20 billion to Apple in 2022 alone to be the exclusive default search engine on Safari."

    class FinancialSchema(BaseModel):
        """Schema for extracting financial and date information."""

        Organization: list[str] = Field(
            description="The name of a company or organization."
        )
        Money: list[str] = Field(
            description="An amount of money, including the currency symbol."
        )
        Date: list[str] = Field(description="A specific date, year, or time period.")

    engine = CoreEngine(model=llm, schema=FinancialSchema)

    result = await engine.run(text, mode="lcel")

    result_dict = dict(result)

    # Check for the monetary value
    assert "$20 billion" in " ".join(result_dict.keys())

    # Check for the date
    assert "2022" in result_dict

    # Check for the organizations
    assert "Google" in result_dict
    assert "Apple" in result_dict
