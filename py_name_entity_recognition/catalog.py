# py_name_entity_recognition/catalog.py
"""
A catalog of predefined NER schemas for scientific and biomedical text.

This module provides a comprehensive, maintainable, and extensible catalog of
Named Entity Recognition (NER) entity definitions tailored for scientific literature.
It uses a registry pattern to store entity definitions and provides functions
to dynamically generate Pydantic models for extraction.
"""

import logging
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field, create_model

# Configure logging
logger = logging.getLogger(__name__)


class EntityDefinition(TypedDict):
    """
    Structure for defining an NER entity.

    Attributes:
        name: The display name of the entity.
        description: A detailed, unambiguous description suitable for LLM extraction prompts,
                     including examples of what to extract and what to avoid.
        category: The logical category of the entity (e.g., "Disorders", "Chemicals").
    """

    name: str
    description: str
    category: str


# ==============================================================================
# == THE CORE ENTITY REGISTRY
# ==============================================================================
# This dictionary maps a unique entity key (e.g., "DiseaseOrSyndrome") to its
# detailed definition. The descriptions are critical for the LLM's performance.

ENTITY_REGISTRY: dict[str, EntityDefinition] = {
    # Category: DISORDERS_AND_FINDINGS
    "DiseaseOrSyndrome": {
        "name": "Disease or Syndrome",
        "description": "Extract specific diseases, disorders, or syndromes. This includes well-defined medical conditions. Examples: 'Type 2 Diabetes', 'Alzheimer's disease', 'COVID-19', 'hypertension'. Do not extract general symptoms unless they are part of a named syndrome.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "NeoplasticProcess": {
        "name": "Neoplastic Process",
        "description": "Extract mentions of cancers, tumors, and malignancies. This includes specific cancer types and general terms for cancerous growths. Examples: 'melanoma', 'glioblastoma', 'adenocarcinoma', 'tumor', 'metastatic cancer'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "SignOrSymptom": {
        "name": "Sign or Symptom",
        "description": "Extract patient-reported symptoms or objective clinical signs. These are observable indicators of a condition. Examples: 'headache', 'fever', 'dizziness', 'shortness of breath', 'elevated blood pressure', 'bradycardia'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "ClinicalFinding": {
        "name": "Clinical Finding",
        "description": "Extract observations derived from clinical tests, examinations, or imaging, excluding simple vital signs. Examples: 'negative result', 'tumor detected on MRI', 'presence of antibodies', 'obstructed artery', 'left ventricular hypertrophy'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "MentalOrBehavioralDysfunction": {
        "name": "Mental or Behavioral Dysfunction",
        "description": "Extract mental health conditions, psychiatric disorders, or significant behavioral issues. Examples: 'major depressive disorder', 'post-traumatic stress disorder (PTSD)', 'anxiety', 'schizophrenia'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "InjuryOrPoisoning": {
        "name": "Injury or Poisoning",
        "description": "Extract mentions of physical injuries, traumas, or poisonings from external causes. Examples: 'concussion', 'fractured femur', 'third-degree burn', 'lead poisoning'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    "PathologicFunction": {
        "name": "Pathologic Function",
        "description": "Extract abnormal physiological processes or states at the cellular or organ level. This describes the 'how' of a disease. Examples: 'inflammation', 'necrosis', 'fibrosis', 'cellular atypia', 'insulin resistance'.",
        "category": "DISORDERS_AND_FINDINGS",
    },
    # Category: CHEMICALS_AND_DRUGS
    "ClinicalDrug": {
        "name": "Clinical Drug",
        "description": "Extract specific marketed drug names, including brand names and formulations. These are typically capitalized. Examples: 'Lipitor 20mg', 'Tylenol', 'Ozempic', 'Advil'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "PharmacologicSubstance": {
        "name": "Pharmacologic Substance",
        "description": "Extract active ingredients (generic names), drug classes, or substances described by their mechanism of action. Examples: 'atorvastatin', 'metformin', 'statins', 'ACE inhibitor', 'monoclonal antibody'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "Antibiotic": {
        "name": "Antibiotic",
        "description": "Extract substances specifically identified as antibiotics, which inhibit or kill bacteria. Examples: 'penicillin', 'amoxicillin', 'doxycycline'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "Vaccine": {
        "name": "Vaccine",
        "description": "Extract preparations that provide active acquired immunity to a particular infectious disease. Examples: 'mRNA-1273', 'Pfizer-BioNTech vaccine', 'BCG vaccine'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "BiologicallyActiveSubstance": {
        "name": "Biologically Active Substance",
        "description": "Extract endogenous or exogenous substances with a biological effect, such as hormones, cytokines, or neurotransmitters. Examples: 'insulin', 'testosterone', 'interferon-gamma', 'dopamine'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "ChemicalEntity": {
        "name": "Chemical Entity",
        "description": "Extract basic chemical compounds, elements, or ions, often discussed in a non-pharmacological context. Examples: 'glucose', 'ethanol', 'sodium chloride', 'iron', 'O2'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "DosageForm": {
        "name": "Dosage Form",
        "description": "Extract the physical form in which a drug is produced and dispensed. Examples: 'tablet', 'injection', 'oral solution', 'transdermal patch', 'inhaler'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    "RouteOfAdministration": {
        "name": "Route of Administration",
        "description": "Extract the path by which a drug, fluid, poison, or other substance is taken into the body. Examples: 'oral', 'intravenous', 'subcutaneous', 'intramuscular', 'topical'.",
        "category": "CHEMICALS_AND_DRUGS",
    },
    # Category: PROCEDURES_AND_INTERVENTIONS
    "TherapeuticProcedure": {
        "name": "Therapeutic Procedure",
        "description": "Extract non-drug treatments or therapies intended to heal or manage a condition. Examples: 'physical therapy', 'psychotherapy', 'counseling', 'radiation therapy'.",
        "category": "PROCEDURES_AND_INTERVENTIONS",
    },
    "SurgicalIntervention": {
        "name": "Surgical Intervention",
        "description": "Extract medical procedures involving incision, manipulation, or suturing of tissue. Examples: 'appendectomy', 'coronary artery bypass grafting', 'biopsy', 'mastectomy'.",
        "category": "PROCEDURES_AND_INTERVENTIONS",
    },
    "DiagnosticProcedure": {
        "name": "Diagnostic Procedure",
        "description": "Extract tests or procedures performed to identify, diagnose, or monitor a condition. Examples: 'MRI scan', 'electrocardiogram (ECG)', 'colonoscopy', 'blood test'.",
        "category": "PROCEDURES_AND_INTERVENTIONS",
    },
    "LaboratoryProcedure": {
        "name": "Laboratory Procedure",
        "description": "Extract techniques and methods used in a laboratory for research or analysis. Examples: 'polymerase chain reaction (PCR)', 'ELISA', 'western blot', 'DNA sequencing'.",
        "category": "PROCEDURES_AND_INTERVENTIONS",
    },
    "MedicalDevice": {
        "name": "Medical Device",
        "description": "Extract instruments, apparatuses, or implants used for medical purposes. Examples: 'pacemaker', 'stent', 'ventilator', 'catheter', 'syringe'.",
        "category": "PROCEDURES_AND_INTERVENTIONS",
    },
    # Category: ANATOMY_AND_PHYSIOLOGY
    "AnatomicalStructure": {
        "name": "Anatomical Structure",
        "description": "Extract organs, body parts, or systems. Examples: 'heart', 'liver', 'central nervous system', 'femur', 'pulmonary artery'.",
        "category": "ANATOMY_AND_PHYSIOLOGY",
    },
    "Tissue": {
        "name": "Tissue",
        "description": "Extract collections of specialized cells that perform a particular function. Examples: 'epithelial tissue', 'muscle tissue', 'connective tissue', 'adipose tissue'.",
        "category": "ANATOMY_AND_PHYSIOLOGY",
    },
    "CellType": {
        "name": "Cell Type",
        "description": "Extract specific types of cells. Examples: 'T-cell', 'neuron', 'hepatocyte', 'erythrocyte', 'stem cell'.",
        "category": "ANATOMY_AND_PHYSIOLOGY",
    },
    "PhysiologicFunction": {
        "name": "Physiologic Function",
        "description": "Extract normal functions and processes of living organisms or their parts. Examples: 'metabolism', 'respiration', 'digestion', 'synaptic transmission', 'glomerular filtration'.",
        "category": "ANATOMY_AND_PHYSIOLOGY",
    },
    # Category: GENETICS_AND_MOLECULAR
    "GeneOrGenome": {
        "name": "Gene or Genome",
        "description": "Extract specific genes, gene families, or genomic regions. Typically alphanumeric. Examples: 'BRCA1', 'TP53', 'Sars-CoV-2 genome', 'human leukocyte antigen (HLA)'.",
        "category": "GENETICS_AND_MOLECULAR",
    },
    "Protein": {
        "name": "Protein",
        "description": "Extract mentions of proteins, peptides, and enzymes. Examples: 'hemoglobin', 'collagen', 'p53 protein', 'caspase-3', 'insulin-like growth factor 1'.",
        "category": "GENETICS_AND_MOLECULAR",
    },
    "GeneticVariant": {
        "name": "Genetic Variant",
        "description": "Extract specific genetic mutations, single nucleotide polymorphisms (SNPs), or other variations. Examples: 'SNP rs12345', 'deletion in exon 9', 'C677T variant'.",
        "category": "GENETICS_AND_MOLECULAR",
    },
    "BiologicalProcess": {
        "name": "Biological Process",
        "description": "Extract recognized series of molecular or cellular events, such as pathways or cycles. Examples: 'apoptosis', 'glycolysis', 'cell division', 'immune response', 'DNA replication'.",
        "category": "GENETICS_AND_MOLECULAR",
    },
    "MolecularFunction": {
        "name": "Molecular Function",
        "description": "Extract activities occurring at the molecular level, such as catalysis or binding. Examples: 'DNA binding', 'kinase activity', 'receptor antagonist', 'ion transport'.",
        "category": "GENETICS_AND_MOLECULAR",
    },
    # Category: EPIDEMIOLOGY_AND_POPULATION
    "PopulationGroup": {
        "name": "Population Group",
        "description": "Extract specific groups of individuals in a study. Examples: 'patients', 'control group', 'elderly participants', 'pediatric population', 'smokers'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    "DemographicCharacteristic": {
        "name": "Demographic Characteristic",
        "description": "Extract characteristics of a study population. Examples: 'age', 'sex', 'race', 'socioeconomic status'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    "RiskFactor": {
        "name": "Risk Factor",
        "description": "Extract variables or exposures associated with an increased risk of a disease or outcome. Examples: 'smoking history', 'obesity', 'hypertension', 'family history of cancer'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    "EnvironmentalExposure": {
        "name": "Environmental Exposure",
        "description": "Extract contact with environmental agents or factors. Examples: 'air pollution', 'pesticide exposure', 'radon gas', 'secondhand smoke'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    "OccupationalExposure": {
        "name": "Occupational Exposure",
        "description": "Extract exposures that occur specifically in the workplace. Examples: 'asbestos exposure', 'exposure to industrial solvents', 'noise exposure in factory workers'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    "Pathogen": {
        "name": "Pathogen",
        "description": "Extract disease-causing microorganisms. Examples: 'Staphylococcus aureus', 'Human Immunodeficiency Virus (HIV)', 'SARS-CoV-2', 'bacteria', 'virus'.",
        "category": "EPIDEMIOLOGY_AND_POPULATION",
    },
    # Category: STUDY_DESIGN_AND_METRICS
    "StudyDesign": {
        "name": "Study Design",
        "description": "Extract the methodology or design of a research study. Examples: 'randomized controlled trial (RCT)', 'cohort study', 'case-control study', 'meta-analysis', 'cross-sectional survey'.",
        "category": "STUDY_DESIGN_AND_METRICS",
    },
    "EpidemiologicalMetric": {
        "name": "Epidemiological Metric",
        "description": "Extract measures of disease frequency, association, or effect. Examples: 'incidence rate', 'prevalence', 'odds ratio (OR)', 'hazard ratio (HR)', 'relative risk (RR)'.",
        "category": "STUDY_DESIGN_AND_METRICS",
    },
    "StatisticalMethod": {
        "name": "Statistical Method",
        "description": "Extract techniques used for data analysis. Examples: 't-test', 'chi-squared test', 'regression analysis', 'ANOVA'.",
        "category": "STUDY_DESIGN_AND_METRICS",
    },
    "StatisticalValue": {
        "name": "Statistical Value",
        "description": "Extract reported statistical findings, often including numbers and symbols. Examples: 'p-value < 0.05', '95% Confidence Interval (CI)', 'p = 0.01', 'OR of 2.5'.",
        "category": "STUDY_DESIGN_AND_METRICS",
    },
    "Bias": {
        "name": "Bias",
        "description": "Extract terms for systematic errors in study design or conduct that can lead to incorrect results. Examples: 'selection bias', 'recall bias', 'confounding', 'publication bias'.",
        "category": "STUDY_DESIGN_AND_METRICS",
    },
    # Category: CLINICAL_TRIAL_SPECIFICS
    "TrialPhase": {
        "name": "Trial Phase",
        "description": "Extract the specific phase of a clinical trial. Examples: 'Phase I', 'Phase II', 'Phase III', 'Phase IV', 'pre-clinical'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "InterventionModel": {
        "name": "Intervention Model",
        "description": "Extract how interventions are assigned to participants in a trial. Examples: 'crossover design', 'parallel group', 'single group assignment'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "BlindingOrMasking": {
        "name": "Blinding or Masking",
        "description": "Extract procedures used to prevent bias by concealing the intervention type from participants, investigators, or assessors. Examples: 'double-blind', 'single-blind', 'open-label', 'unblinded'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "PrimaryOutcome": {
        "name": "Primary Outcome",
        "description": "Extract the main result that is measured at the end of a study to see if a given treatment worked. Examples: 'overall survival', 'reduction in tumor size', 'change in blood pressure'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "SecondaryOutcome": {
        "name": "Secondary Outcome",
        "description": "Extract additional outcomes that are measured to help evaluate the effects of an intervention. Examples: 'quality of life', 'rate of adverse events', 'disease-free survival'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "AdverseEvent": {
        "name": "Adverse Event (AE)",
        "description": "Extract any untoward medical occurrence in a patient or clinical investigation subject administered a pharmaceutical product, which does not necessarily have a causal relationship with this treatment. Examples: 'nausea', 'headache', 'rash'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "SeriousAdverseEvent": {
        "name": "Serious Adverse Event (SAE)",
        "description": "Extract adverse events that result in death, are life-threatening, require inpatient hospitalization, or result in persistent or significant disability. Examples: 'myocardial infarction', 'anaphylactic shock', 'hospitalization for pneumonia'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "EligibilityCriteria": {
        "name": "Eligibility Criteria",
        "description": "Extract general requirements that potential participants must meet to be included in a study. Look for terms like 'inclusion criteria' or 'exclusion criteria'. Examples: 'adults over 18', 'patients with confirmed diagnosis', 'non-smokers'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    "SampleSize": {
        "name": "Sample Size",
        "description": "Extract the number of participants enrolled in a study. Look for explicit mentions of the sample size. Examples: 'a total of 500 patients', 'N=250', 'sample of 100 individuals'.",
        "category": "CLINICAL_TRIAL_SPECIFICS",
    },
    # Category: ORGANIZATIONS_AND_CONTEXT
    "PharmaceuticalCompany": {
        "name": "Pharmaceutical Company",
        "description": "Extract companies involved in the development, manufacturing, and sale of drugs. Examples: 'Pfizer Inc.', 'AstraZeneca', 'Roche'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    "ResearchInstitution": {
        "name": "Research Institution",
        "description": "Extract universities, hospitals, or centers where research is conducted. Examples: 'Mayo Clinic', 'Harvard University', 'National Institutes of Health (NIH)'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    "RegulatoryAgency": {
        "name": "Regulatory Agency",
        "description": "Extract governmental bodies that regulate medical products and research. Examples: 'Food and Drug Administration (FDA)', 'European Medicines Agency (EMA)'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    "FundingSource": {
        "name": "Funding Source",
        "description": "Extract organizations that provide financial support for the research. Examples: 'funded by the NIH', 'supported by a grant from the Wellcome Trust'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    "Location": {
        "name": "Location",
        "description": "Extract geographic locations such as cities, states, or countries relevant to the study context. Examples: 'Boston, Massachusetts', 'conducted in Europe', 'a multi-center trial in China'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    "Person": {
        "name": "Person",
        "description": "Extract names of specific individuals, such as researchers, authors, or principal investigators. Examples: 'Dr. John Smith', 'the research team of Jane Doe'.",
        "category": "ORGANIZATIONS_AND_CONTEXT",
    },
    # Category: VETERINARY_MEDICINE
    "AnimalSpecies": {
        "name": "Animal Species",
        "description": "Extract the species of animal being studied, including common names and scientific names. Examples: 'mice', 'Felis catus', 'bovine', 'zebrafish'.",
        "category": "VETERINARY_MEDICINE",
    },
    "VeterinaryDrug": {
        "name": "Veterinary Drug",
        "description": "Extract drugs, vaccines, or therapeutic substances specifically formulated or used for animals. Examples: 'Ivermectin', 'Rimadyl', 'Bravecto', 'Feline leukemia vaccine'.",
        "category": "VETERINARY_MEDICINE",
    },
    "AnimalDisease": {
        "name": "Animal Disease",
        "description": "Extract diseases, syndromes, or conditions that primarily affect animals. Examples: 'canine parvovirus', 'foot-and-mouth disease', 'avian influenza', 'heartworm disease'.",
        "category": "VETERINARY_MEDICINE",
    },
    "VeterinaryProcedure": {
        "name": "Veterinary Procedure",
        "description": "Extract diagnostic, therapeutic, or surgical procedures performed on animals. Examples: 'spaying', 'dehorning', 'equine lameness examination', 'necropsy'.",
        "category": "VETERINARY_MEDICINE",
    },
    # Category: HEALTHCARE_ECONOMICS_AND_POLICY
    "HealthcareCost": {
        "name": "Healthcare Cost",
        "description": "Extract mentions of monetary costs, expenses, or economic values related to healthcare services, drugs, or equipment. Examples: '$50 per treatment', 'hospitalization costs of $10,000', 'insurance reimbursement'.",
        "category": "HEALTHCARE_ECONOMICS_AND_POLICY",
    },
    "EconomicOutcome": {
        "name": "Economic Outcome",
        "description": "Extract metrics used in health economics to evaluate the value or efficiency of an intervention. Examples: 'Quality-Adjusted Life-Year (QALY)', 'cost-benefit ratio', 'incremental cost-effectiveness ratio (ICER)'.",
        "category": "HEALTHCARE_ECONOMICS_AND_POLICY",
    },
    "HealthPolicy": {
        "name": "Health Policy",
        "description": "Extract specific laws, regulations, or official guidelines from government or health organizations that affect healthcare. Examples: 'Medicare policy', 'the Affordable Care Act', 'vaccination mandate'.",
        "category": "HEALTHCARE_ECONOMICS_AND_POLICY",
    },
    "InsuranceProvider": {
        "name": "Insurance Provider",
        "description": "Extract names of health insurance companies or public payers. Examples: 'Blue Cross Blue Shield', 'UnitedHealthcare', 'Medicaid', 'NHS'.",
        "category": "HEALTHCARE_ECONOMICS_AND_POLICY",
    },
    # Category: PUBLIC_HEALTH_AND_SYSTEMS
    "PublicHealthIntervention": {
        "name": "Public Health Intervention",
        "description": "Extract programs or strategies aimed at improving the health of a population. Examples: 'smoking cessation campaign', 'water fluoridation', 'nationwide screening program', 'public health advisory'.",
        "category": "PUBLIC_HEALTH_AND_SYSTEMS",
    },
    "HealthSystem": {
        "name": "Health System",
        "description": "Extract terms describing the organization of people, institutions, and resources that deliver health care services. Examples: 'National Health Service (NHS)', 'single-payer system', 'integrated delivery network'.",
        "category": "PUBLIC_HEALTH_AND_SYSTEMS",
    },
    "CareGuideline": {
        "name": "Care Guideline",
        "description": "Extract official recommendations for the treatment or management of a condition. Examples: 'ACC/AHA guidelines', 'NICE guidelines', 'standard treatment protocol'.",
        "category": "PUBLIC_HEALTH_AND_SYSTEMS",
    },
    "HealthDisparity": {
        "name": "Health Disparity",
        "description": "Extract mentions of differences in health outcomes between groups of people. Examples: 'health outcomes by income', 'racial health gap', 'disparities in access to care'.",
        "category": "PUBLIC_HEALTH_AND_SYSTEMS",
    },
    # Category: BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY
    "SoftwareTool": {
        "name": "Software Tool",
        "description": "Extract names of specific software, packages, or tools used for data analysis in biomedical research. Examples: 'BLAST', 'GATK', 'R package Seurat', 'ImageJ'.",
        "category": "BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY",
    },
    "BiologicalDatabase": {
        "name": "Biological Database",
        "description": "Extract names of databases or repositories storing biological data. Examples: 'GenBank', 'Protein Data Bank (PDB)', 'The Cancer Genome Atlas (TCGA)', 'UniProt'.",
        "category": "BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY",
    },
    "Algorithm": {
        "name": "Algorithm",
        "description": "Extract names of specific algorithms used for computational analysis. Examples: 'Smith-Waterman algorithm', 'UMAP', 't-SNE', 'Bowtie algorithm'.",
        "category": "BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY",
    },
    "DataFormat": {
        "name": "Data Format",
        "description": "Extract file formats or data structures used in bioinformatics. Examples: 'FASTA format', 'VCF file', 'BAM', 'FASTQ'.",
        "category": "BIOINFORMATICS_AND_COMPUTATIONAL_BIOLOGY",
    },
}


# ==============================================================================
# == PRESETS
# ==============================================================================
# Predefined collections of entity keys for common use cases.

PRESETS: dict[str, list[str]] = {
    "CLINICAL_TRIAL_CORE": [
        "DiseaseOrSyndrome",
        "ClinicalDrug",
        "PharmacologicSubstance",
        "TherapeuticProcedure",
        "MedicalDevice",
        "PopulationGroup",
        "StudyDesign",
        "TrialPhase",
        "BlindingOrMasking",
        "PrimaryOutcome",
        "SecondaryOutcome",
        "AdverseEvent",
        "SeriousAdverseEvent",
        "EligibilityCriteria",
        "SampleSize",
    ],
    "EPIDEMIOLOGY_FOCUS": [
        "DiseaseOrSyndrome",
        "SignOrSymptom",
        "PopulationGroup",
        "DemographicCharacteristic",
        "RiskFactor",
        "EnvironmentalExposure",
        "OccupationalExposure",
        "Pathogen",
        "StudyDesign",
        "EpidemiologicalMetric",
        "StatisticalValue",
        "Bias",
        "Location",
    ],
    "PHARMACOVIGILANCE": [
        "ClinicalDrug",
        "PharmacologicSubstance",
        "DosageForm",
        "RouteOfAdministration",
        "AdverseEvent",
        "SeriousAdverseEvent",
        "DiseaseOrSyndrome",
        "SignOrSymptom",
        "PopulationGroup",
    ],
    "MOLECULAR_BIOLOGY": [
        "GeneOrGenome",
        "Protein",
        "GeneticVariant",
        "CellType",
        "Tissue",
        "BiologicalProcess",
        "MolecularFunction",
        "LaboratoryProcedure",
        "PathologicFunction",
    ],
    "VETERINARY_RESEARCH": [
        "AnimalSpecies",
        "VeterinaryDrug",
        "AnimalDisease",
        "VeterinaryProcedure",
        "PopulationGroup",
        "StudyDesign",
    ],
    "HEALTH_ECONOMICS": [
        "DiseaseOrSyndrome",
        "ClinicalDrug",
        "TherapeuticProcedure",
        "HealthcareCost",
        "EconomicOutcome",
        "HealthPolicy",
        "InsuranceProvider",
        "PopulationGroup",
    ],
    "PUBLIC_HEALTH": [
        "DiseaseOrSyndrome",
        "PublicHealthIntervention",
        "HealthSystem",
        "CareGuideline",
        "HealthDisparity",
        "PopulationGroup",
        "DemographicCharacteristic",
        "EpidemiologicalMetric",
        "Location",
    ],
    "BIOINFORMATICS": [
        "GeneOrGenome",
        "Protein",
        "GeneticVariant",
        "SoftwareTool",
        "BiologicalDatabase",
        "Algorithm",
        "DataFormat",
        "LaboratoryProcedure",
    ],
    "COMPREHENSIVE": list(ENTITY_REGISTRY.keys()),
}


# ==============================================================================
# == DYNAMIC FUNCTIONS
# ==============================================================================


def register_entity(
    key: str, definition: EntityDefinition, overwrite: bool = False
) -> None:
    """
    Adds or updates an entity definition in the central registry at runtime.

    Args:
        key: The unique key for the entity (e.g., "CustomBiomarker").
        definition: An EntityDefinition dictionary containing the entity's metadata.
        overwrite: If True, allows overwriting an existing entity with the same key.
                   Defaults to False.

    Raises:
        ValueError: If the key already exists and 'overwrite' is False.
    """
    if key in ENTITY_REGISTRY and not overwrite:
        raise ValueError(
            f"Entity key '{key}' already exists in the registry. "
            "Set overwrite=True to replace it."
        )
    ENTITY_REGISTRY[key] = definition
    logger.info(f"Entity '{key}' has been registered.")


def _generate_pydantic_model(
    model_name: str, description: str, entity_keys: set[str]
) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic BaseModel from a set of entity keys.

    Args:
        model_name: The name for the created Pydantic model class.
        description: The docstring for the created Pydantic model.
        entity_keys: A set of keys from the ENTITY_REGISTRY to include as fields.

    Returns:
        A new Pydantic BaseModel class with the specified fields.
    """
    fields: dict[str, Any] = {}
    for key in sorted(entity_keys):  # Sort for consistent model definition
        if key in ENTITY_REGISTRY:
            entity_def = ENTITY_REGISTRY[key]
            # Use the key as the field name. Pydantic fields should be valid identifiers.
            field_name = key
            # The type hint must be List[str] for multi-value extraction.
            # The description from the registry is used to guide the LLM.
            # default_factory=list makes extraction of the field optional.
            fields[field_name] = (
                list[str],
                Field(default_factory=list, description=entity_def["description"]),
            )
        else:
            logger.warning(
                f"Entity key '{key}' not found in registry and will be skipped."
            )

    if not fields:
        raise ValueError(
            "Cannot generate a Pydantic model with no fields. Check your entity keys."
        )

    # Use pydantic.create_model to dynamically construct the BaseModel
    dynamic_model = create_model(
        model_name,
        __doc__=description,
        **fields,
    )
    return dynamic_model


def get_schema(
    preset: Optional[str] = None,
    include_categories: Optional[list[str]] = None,
    include_entities: Optional[list[str]] = None,
    exclude_entities: Optional[list[str]] = None,
    schema_name: str = "CustomNERSchema",
    schema_description: str = "A dynamically generated Pydantic model for scientific NER.",
) -> type[BaseModel]:
    """
    Retrieves a dynamically generated Pydantic schema based on specified criteria.

    This function builds a set of desired entities from presets, categories, and
    explicit inclusions, then removes any exclusions, and finally generates a
    Pydantic model from the resulting set.

    Args:
        preset: The name of a predefined preset (e.g., "CLINICAL_TRIAL_CORE").
        include_categories: A list of category names. All entities in these categories
                            will be included.
        include_entities: A list of specific entity keys to include.
        exclude_entities: A list of specific entity keys to exclude from the final set.
        schema_name: The class name for the generated Pydantic model.
        schema_description: The docstring for the generated Pydantic model.

    Returns:
        A Pydantic BaseModel class configured with the selected NER fields.

    Raises:
        ValueError: If a specified preset is not found or if the final set of
                    entities is empty.
    """
    selected_entity_keys: set[str] = set()

    # 1. Add entities from a preset
    if preset:
        preset_upper = preset.upper()
        if preset_upper not in PRESETS:
            raise ValueError(
                f"Preset '{preset}' not found. Available presets: {list(PRESETS.keys())}"
            )
        selected_entity_keys.update(PRESETS[preset_upper])
        logger.info(
            f"Loaded {len(PRESETS[preset_upper])} entities from preset: {preset}"
        )

    # 2. Add entities from included categories
    if include_categories:
        cat_keys = {
            key
            for key, definition in ENTITY_REGISTRY.items()
            if definition["category"] in include_categories
        }
        selected_entity_keys.update(cat_keys)
        logger.info(
            f"Added {len(cat_keys)} entities from categories: {include_categories}"
        )

    # 3. Add specific entities
    if include_entities:
        # Validate that all included entities exist in the registry
        unknown_entities = set(include_entities) - set(ENTITY_REGISTRY.keys())
        if unknown_entities:
            raise ValueError(
                "The following entities from 'include_entities' are not in the registry: "
                f"{', '.join(sorted(unknown_entities))}"  # Sort for consistent error messages
            )
        selected_entity_keys.update(include_entities)
        logger.info(f"Included {len(include_entities)} specific entities.")

    # 4. If no selections were made, default to comprehensive
    if not selected_entity_keys:
        logger.warning(
            "No preset, categories, or entities specified. Defaulting to 'COMPREHENSIVE' preset."
        )
        selected_entity_keys.update(PRESETS["COMPREHENSIVE"])

    # 5. Remove any excluded entities
    if exclude_entities:
        excluded_set = set(exclude_entities)
        original_count = len(selected_entity_keys)
        selected_entity_keys -= excluded_set
        logger.info(f"Excluded {original_count - len(selected_entity_keys)} entities.")

    # 6. Check if the final set is empty
    if not selected_entity_keys:
        raise ValueError(
            "The combination of inclusions and exclusions resulted in an empty set of entities. "
            "Cannot generate an empty schema."
        )

    # 7. Generate the Pydantic model
    logger.info(
        f"Generating schema '{schema_name}' with {len(selected_entity_keys)} total fields."
    )
    return _generate_pydantic_model(
        model_name=schema_name,
        description=schema_description,
        entity_keys=selected_entity_keys,
    )
