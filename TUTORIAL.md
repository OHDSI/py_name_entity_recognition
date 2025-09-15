# Tutorial: Using the Scientific NER Catalog

Welcome! This tutorial is a deep dive into one of the most powerful features of the `pyNameEntityRecognition` package: the **Scientific NER Catalog**.

While the main `README.md` and `docs/vignette.md` show how to define your own NER schema from scratch, the catalog provides a massive, pre-built library of entity definitions specifically tailored for complex domains like clinical trials, molecular biology, and epidemiology.

Using the catalog allows you to:
- **Get started faster**: No need to manually define dozens of entities.
- **Improve accuracy**: The catalog's entity descriptions are carefully crafted to guide LLMs effectively.
- **Ensure consistency**: Use standardized, community-vetted entity definitions in your projects.

This guide will walk you through everything from using pre-built **presets** to dynamically **customizing** a schema to fit your exact needs.

Let's get started!

---

## 1. The Easiest Way: Using Presets

Presets are pre-defined collections of entities for common NER tasks. They are the quickest way to get up and running with a powerful, domain-specific schema.

Let's say you're working with a clinical trial abstract. Instead of defining entities like "TrialPhase," "AdverseEvent," and "PrimaryOutcome" yourself, you can just use the `CLINICAL_TRIAL_CORE` preset.

### Example: Extracting Clinical Trial Information

Here’s how you would use the `extract_entities` function with a preset. Notice that instead of passing a Pydantic class to the `schema` argument, we just pass the preset's name as a string.

```python
import asyncio
from py_name_entity_recognition import extract_entities
from py_name_entity_recognition.observability.visualization import display_biores

# Example text from a clinical trial abstract
text = """In this randomized controlled trial (RCT), patients with hypertension receiving Metformin
were monitored for adverse events, including nausea and vomiting. The study was conducted in Phase III."""

async def main():
    print("--- Using Preset: CLINICAL_TRIAL_CORE ---")

    # Just pass the preset name to the `schema` parameter
    results = await extract_entities(
        input_data=text,
        schema="CLINICAL_TRIAL_CORE"
    )

    # Display the results in a nice, color-coded format
    display_biores(results)

if __name__ == "__main__":
    # To run this, you need a configured LLM (e.g., set OPENAI_API_KEY)
    # asyncio.run(main())
    print("Example ready to run with a configured LLM.")
```

This will extract entities relevant to a clinical trial, such as "randomized controlled trial" (StudyDesign), "hypertension" (DiseaseOrSyndrome), "Metformin" (PharmacologicSubstance), "adverse events" (AdverseEvent), and "Phase III" (TrialPhase).

### Available Presets

The catalog comes with a variety of presets for different scientific domains. Here is the complete list:

- `CLINICAL_TRIAL_CORE`: Key entities for clinical trial documents.
- `EPIDEMIOLOGY_FOCUS`: Entities related to population studies and public health.
- `PHARMACOVIGILANCE`: For monitoring adverse drug events.
- `MOLECULAR_BIOLOGY`: Genes, proteins, and other molecular-level entities.
- `VETERINARY_RESEARCH`: For studies involving animals.
- `HEALTH_ECONOMICS`: Costs, policies, and economic outcomes in healthcare.
- `PUBLIC_HEALTH`: Interventions, systems, and disparities in public health.
- `BIOINFORMATICS`: Software tools, databases, and algorithms.
- `COMPREHENSIVE`: Includes **all** entities in the catalog for maximum coverage.

---

## 2. Total Control: Customizing Your Schema

Presets are great, but what if you need a specific combination of entities? The catalog gives you full control to build a custom schema by mixing and matching categories and individual entities.

Instead of a string, you pass a dictionary to the `schema` parameter with keys like `include_categories`, `include_entities`, and `exclude_entities`.

### Method 1: Building a Schema from Categories

You can create a schema by picking broad categories. For example, if you're interested in drugs and their effects, you could include the `CHEMICALS_AND_DRUGS` and `DISORDERS_AND_FINDINGS` categories.

```python
import asyncio
from py_name_entity_recognition import extract_entities

text = """In this randomized controlled trial (RCT), patients with hypertension receiving Metformin
were monitored for adverse events, including nausea and vomiting. The study was conducted in Phase III."""

async def main():
    print("\n--- Using a Dynamic Schema from Categories ---")

    # Define the schema using a dictionary
    config = {
        "include_categories": [
            "CHEMICALS_AND_DRUGS",
            "DISORDERS_AND_FINDINGS"
        ]
    }

    results = await extract_entities(
        input_data=text,
        schema=config,
        output_format="json"  # Let's get JSON this time
    )

    print(results)

if __name__ == "__main__":
    # asyncio.run(main())
    print("Example ready to run with a configured LLM.")
```

This would extract "hypertension" (from `DISORDERS_AND_FINDINGS`) and "Metformin" (from `CHEMICALS_AND_DRUGS`), but ignore "Phase III" because the `CLINICAL_TRIAL_SPECIFICS` category was not included.

### Method 2: Fine-Tuning with Includes and Excludes

This is the most powerful method. You can start with a broad base (like a preset or a category) and then refine it by adding or removing specific entities.

Let's start with the `PHARMACOVIGILANCE` preset but exclude the `DosageForm` entity because we don't need it.

```python
# (imports and text are the same as above)

async def main():
    print("\n--- Fine-Tuning a Preset ---")

    config = {
        "preset": "PHARMACOVIGILANCE",  # Start with a preset
        "exclude_entities": ["DosageForm", "RouteOfAdministration"] # But remove what we don't need
    }

    results = await extract_entities(
        input_data=text,
        schema=config,
        output_format="json"
    )

    print(results)

if __name__ == "__main__":
    # asyncio.run(main())
    print("Example ready to run with a configured LLM.")
```
You can also use `include_entities` to add specific entities to a preset or category-based schema.

### Available Entity Categories

To help you build your custom schemas, here is the full list of available entity categories:

- `DISORDERS_AND_FINDINGS`
- `CHEMICALS_AND_DRUGS`
- `PROCEDURES_AND_INTERVENTIONS`
- `ANATOMY_AND_PHYSIOLOGY`
- `GENETICS_AND_MOLECULAR`
- `EPIDEMIOLOGY_AND_POPULATION`
- `STUDY_DESIGN_AND_METRICS`
- `CLINICAL_TRIAL_SPECIFICS`
- `ORGANIZATIONS_AND_CONTEXT`
- `VETERINARY_MEDICINE`
- `HEALTHCARE_ECONOMICS_AND_POLICY`
- `PUBLIC_HEALTH_AND_SYSTEMS`
- `BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY`

---

## 3. Advanced Usage: Extending the Catalog

For ultimate flexibility, you can even add your own entity definitions to the catalog at runtime using the `register_entity` function. This is useful if you have a very specific, custom entity type that you want to reuse across different schemas.

```python
from py_name_entity_recognition.catalog import register_entity, get_schema
from py_name_entity_recognition import extract_entities

# Define a new entity
register_entity(
    key="CustomBiomarker",
    definition={
        "name": "Custom Biomarker",
        "description": "A specific biomarker relevant to our internal research.",
        "category": "GENETICS_AND_MOLECULAR",
    }
)

# Now you can use it in a custom schema
config = {
    "include_entities": ["CustomBiomarker", "GeneOrGenome"]
}

# The `extract_entities` function will now recognize "CustomBiomarker"
# text = "The study measured levels of NeuroMarker-X."
# results = await extract_entities(text, schema=config)
```

---

## 4. Combining the Catalog with Other Features

Remember, using the catalog doesn't lock you out of other features. You can combine a catalog-based schema with any other parameter of the `extract_entities` function.

### Example: Agentic Mode with a Custom Schema

Want the high accuracy of the `agentic` mode while using a custom schema built from the catalog? No problem.

```python
import asyncio
from py_name_entity_recognition import extract_entities

text = """In this randomized controlled trial (RCT), patients with hypertension receiving Metformin
were monitored for adverse events, including nausea and vomiting. The study was conducted in Phase III."""

async def main():
    print("\n--- Agentic Mode with a Custom Catalog Schema ---")

    config = {
        "preset": "CLINICAL_TRIAL_CORE",
        "exclude_entities": ["EligibilityCriteria", "SampleSize"]
    }

    results = await extract_entities(
        input_data=text,
        schema=config,
        mode="agentic", # Use the self-correcting agent
        output_format="json"
    )

    print(results)

if __name__ == "__main__":
    # asyncio.run(main())
    print("Example ready to run with a configured LLM.")
```

You can still change the `model_config`, process multiple documents in parallel, and let the engine handle long text via automatic chunking, all while using the power and convenience of the Scientific NER Catalog.

---

## Conclusion

This tutorial has demonstrated the power and flexibility of the Scientific NER Catalog. You have learned how to:

- **Use presets** for rapid, domain-specific entity extraction.
- **Build custom schemas** by including categories or specific entities.
- **Fine-tune schemas** by excluding entities you don't need.
- **Extend the catalog** with your own definitions at runtime.
- **Combine the catalog** with all other features of the `pyNameEntityRecognition` package.

You are now ready to tackle complex NER tasks in scientific and biomedical text with confidence. Happy extracting!
