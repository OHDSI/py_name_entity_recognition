# Example Usage: pyNameEntityRecognition

This document demonstrates how to use the `pyNameEntityRecognition` package to extract named entities from text, showcasing both the new catalog-based functionality and the existing custom schema approach.

## Setup

First, import the necessary components and initialize the main extraction function. For these examples, we assume you have a model configuration ready.

```python
# main_example.py
import asyncio
from py_name_entity_recognition import extract_entities
from pydantic import BaseModel, Field
from typing import List

# Example text from a clinical trial abstract
text = """In this randomized controlled trial (RCT), patients with hypertension receiving Metformin
were monitored for adverse events, including nausea and vomiting. The study was conducted in Phase III."""

# --- New Functionality: Using the Catalog ---

async def run_examples():
    # Usage 1: Using a predefined preset string
    print("--- Using Preset: CLINICAL_TRIAL_CORE ---")
    results_preset = await extract_entities(text, schema="CLINICAL_TRIAL_CORE", output_format="json")
    print(results_preset)
    # Note: For actual model_dump_json, you would need the Pydantic object,
    # but extract_entities returns a dict here.

    # Usage 2: Using a dynamic configuration dictionary
    print("\n--- Using Dynamic Config ---")
    config = {
        "include_categories": ["CHEMICALS_AND_DRUGS", "DISORDERS_AND_FINDINGS", "CLINICAL_TRIAL_SPECIFICS"],
        "exclude_entities": ["Antibiotic"]
    }
    results_dynamic = await extract_entities(text, schema=config, output_format="json")
    print(results_dynamic)

    # --- Existing Functionality: Backward Compatibility ---

    # Usage 3: Using a custom Pydantic model
    print("\n--- Using Custom Pydantic Model ---")

    class CustomSchema(BaseModel):
        """A custom schema to extract specific drugs and conditions."""
        Drug: List[str] = Field(description="The drug mentioned in the text.")
        Condition: List[str] = Field(description="The medical condition mentioned.")

    results_custom = await extract_entities(text, schema=CustomSchema, output_format="json")
    print(results_custom)

if __name__ == "__main__":
    # Note: Running these examples requires a configured LLM.
    # The output will depend on the model's ability to extract the entities.
    # To run this example, you might need to set up model credentials, e.g., an OpenAI API key.
    # asyncio.run(run_examples())
    print("Example script structure is set up. To run, configure an LLM and uncomment the asyncio.run call.")

```

### How to Run

To run the examples above, you would save the code as a Python file (e.g., `main_example.py`) and ensure you have an appropriate language model configured (e.g., by setting environment variables for your chosen LLM provider). Then, you can execute the script:

```sh
# Example: Ensure your environment is set up for the LLM
# export OPENAI_API_KEY="your_key_here"

python main_example.py
```

The output will show the extracted entities for each of the three methods, demonstrating the flexibility of the `extract_entities` function.
