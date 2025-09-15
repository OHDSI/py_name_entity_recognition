# Getting Started with pyNameEntityRecognition

Welcome to the `pyNameEntityRecognition` vignette! This guide will walk you through the package's features, from basic usage to more advanced workflows. We'll show you how to extract named entities from text using the power of Large Language Models (LLMs) in a flexible and robust way.

## 1. A Quick Introduction

`pyNameEntityRecognition` is a Python package for LLM-based Named Entity Recognition (NER). It uses LangChain for interacting with LLMs and LangGraph to create powerful, self-correcting extraction workflows.

**Key Features:**

- **Flexible Extraction**: Choose between a fast `lcel` mode and a highly accurate, self-correcting `agentic` mode.
- **Dynamic Schemas**: Define what you want to extract using simple Pydantic models.
- **LLM Agnostic**: Works with OpenAI, Anthropic, Google, and local models via Ollama.
- **Automatic Handling of Long Texts**: The package automatically chunks long documents and merges the results.

## 2. Installation and Setup

First, make sure you have the package and its dependencies installed.

```bash
# Install the package from your repository root
pip install .

# Download the necessary spaCy model for text processing
python -m spacy download en_core_web_sm
```

You'll also need to configure your LLM provider's API keys. Create a `.env` file in your project's root directory:

```
# .env
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
# Add other keys as needed
```

## 3. Basic Usage: Your First Extraction

Let's start with a simple example. We'll extract names, locations, and companies from a sentence.

### Step 3.1: Define Your Extraction Schema

First, you define the structure of the data you want to extract using a Pydantic model. This tells the engine what to look for.

```python
from pydantic import BaseModel, Field
from typing import List

class UserInfo(BaseModel):
    """Schema for extracting user information."""
    Person: List[str] = Field(description="The full name of a person.")
    Location: List[str] = Field(description="A city, state, or country.")
    Company: List[str] = Field(description="The name of a company or organization.")
```

### Step 3.2: Run the Extraction

Now, let's import the main `extract_entities` function and run it on our text.

```python
import asyncio
from py_name_entity_recognition import extract_entities

# The text we want to process
text = "John Doe, a software engineer at Google, lives in New York. He is meeting with Jane Smith from Microsoft tomorrow."

# Run the async extraction function
async def main():
    conll_output = await extract_entities(
        input_data=text,
        schema=UserInfo
    )
    print(conll_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3.3: Understanding the Output

The default output is in a "CoNLL-like" format using BIO-ES tags, which is common in NER tasks. It's a list of `(token, tag)` tuples.

- `B-` stands for the **beginning** of an entity.
- `I-` stands for **inside** an entity.
- `O` stands for **outside** (not an entity).
- `E-` stands for the **end** of an entity.
- `S-` stands for a **single**-token entity.

Here’s what the output for our example looks like:

```
[('John', 'B-PERSON'), ('Doe', 'E-PERSON'), (',', 'O'), ('a', 'O'), ('software', 'O'), ('engineer', 'O'), ('at', 'O'), ('Google', 'S-COMPANY'), (',', 'O'), ('lives', 'O'), ('in', 'O'), ('New', 'B-LOCATION'), ('York', 'E-LOCATION'), ('.', 'O'), ('He', 'O'), ('is', 'O'), ('meeting', 'O'), ('with', 'O'), ('Jane', 'B-PERSON'), ('Smith', 'E-PERSON'), ('from', 'O'), ('Microsoft', 'S-COMPANY'), ('tomorrow', 'O'), ('.', 'O')]
```

### Step 3.4: Visualizing the Results

For a more human-readable view, especially in environments like Jupyter notebooks, you can use the built-in `display_biores` utility.

```python
from py_name_entity_recognition.observability.visualization import display_biores

# Assuming conll_output from the previous step
display_biores(conll_output)
```

This will render the text with color-coded entity spans, similar to spaCy's `displaCy` visualizer.

## 4. Intermediate Usage

Now let's explore some more powerful features.

### 4.1. Getting Structured JSON Output

Instead of CoNLL tags, you might prefer a clean JSON object. Just change the `output_format` parameter.

```python
json_output = await extract_entities(
    input_data=text,
    schema=UserInfo,
    output_format="json"  # <-- Change this
)
print(json_output)
```

The output will be a dictionary that mirrors your Pydantic schema:

```json
{
  "entities": [
    { "type": "Person", "text": "John Doe" },
    { "type": "Company", "text": "Google" },
    { "type": "Location", "text": "New York" },
    { "type": "Person", "text": "Jane Smith" },
    { "type": "Company", "text": "Microsoft" }
  ]
}
```

### 4.2. Switching LLM Providers

The default model is OpenAI's `gpt-3.5-turbo`. You can easily switch to another provider, like Anthropic's Claude, by passing a `model_config`.

```python
from py_name_entity_recognition.models.config import ModelConfig

# Define the model configuration
claude_config = ModelConfig(
    provider="anthropic",
    name="claude-3-sonnet-20240229"
)

# Run extraction with the new model
claude_output = await extract_entities(
    input_data=text,
    schema=UserInfo,
    model_config=claude_config
)
```

### 4.3. Using the Agentic Mode for Higher Accuracy

For challenging texts, the `agentic` mode provides a self-correcting workflow. It first extracts entities, then validates them to ensure they exist in the source text. If it finds "hallucinated" entities, it forces the LLM to refine its output.

This is great for reducing errors and improving the reliability of your extractions.

```python
agentic_output = await extract_entities(
    input_data=text,
    schema=UserInfo,
    mode="agentic"  # <-- The only change needed!
)

display_biores(agentic_output)
```

While slightly slower, this mode is recommended for production use cases where accuracy is critical.

## 5. Advanced Usage

### 5.1. Processing Multiple Documents

`extract_entities` can process a list of texts in a single call, running them in parallel for efficiency.

```python
texts = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Elon Musk announced that Tesla's headquarters will move to Austin."
]

results = await extract_entities(
    input_data=texts,
    schema=UserInfo,
    output_format="json"
)

# `results` will be a list of JSON objects, one for each text.
print(results)
```

### 5.2. Handling Long Documents

What if your text is longer than the LLM's context window? `pyNameEntityRecognition` handles this automatically! It intelligently splits the text into overlapping chunks, processes each chunk, and then merges the results back together.

For most cases, you don't need to do anything. If you need more control, you can instantiate the `CoreEngine` directly to adjust the chunking parameters.

```python
from py_name_entity_recognition.core.engine import CoreEngine
from py_name_entity_recognition.models.factory import ModelFactory

# For very long text...
long_text = "..." * 5000

# You can customize chunking behavior if needed
engine = CoreEngine(
    model=ModelFactory.create(), # Default OpenAI model
    schema=UserInfo,
    chunk_size=3000,       # Max characters per chunk
    chunk_overlap=400      # Characters to overlap between chunks
)

# Run the engine directly
long_doc_output = await engine.run(long_text)
display_biores(long_doc_output)
```

## Conclusion

This vignette covered the core functionalities of `pyNameEntityRecognition`. You learned how to:
- Define an extraction schema.
- Run extractions in both `lcel` and `agentic` modes.
- Change the output format and LLM provider.
- Process single, multiple, and long documents.

You are now ready to start using `pyNameEntityRecognition` for your own NER tasks!
