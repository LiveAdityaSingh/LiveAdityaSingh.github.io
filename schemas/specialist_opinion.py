"""Specialist opinion schema — shared output contract for all specialist agents.

Defines the structured output that every specialist (Cardiology, Neurology,
Orthopedics, Surgery) must produce. This schema feeds directly into:
  - GP Orchestrator (M4): uses `confidence` for weighting
  - Weighted Consolidator (M5): aggregates multiple opinions
  - Consensus Auditor (M6): cross-references findings

All downstream milestones depend on these exact field names.
"""

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single clinical finding from the specialist's analysis.

    Each finding must be traceable to a specific data source (e.g., a DSP
    feature, a lab value, an imaging report) for auditability.
    """

    description: str = Field(
        ..., description="Human-readable description of the finding"
    )
    evidence_source: str = Field(
        ...,
        description=(
            "Where this finding came from: a DSP feature name, "
            "observation code, imaging report section, etc."
        ),
    )
    clinical_significance: str = Field(
        ...,
        description="Clinical interpretation of this finding",
    )


class SpecialistOpinion(BaseModel):
    """Structured output produced by every specialist agent.

    This is the contract between specialist agents and the downstream
    consolidator/auditor pipeline. Every field is used by at least one
    downstream milestone.
    """

    case_id: str = Field(
        ..., description="The patient case this opinion refers to"
    )
    specialty: str = Field(
        ...,
        description="The specialty that produced this opinion (e.g. 'cardiology')",
    )
    diagnosis_codes: list[str] = Field(
        ...,
        description=(
            "ICD-10 diagnosis codes for the identified condition(s). "
            "Use standard ICD-10-CM format (e.g. 'I21.0' for acute STEMI)"
        ),
    )
    snomed_codes: list[str] = Field(
        default_factory=list,
        description=(
            "Optional SNOMED CT codes for UK Core FHIR compatibility. "
            "Maps to the same conditions as diagnosis_codes"
        ),
    )
    diagnosis_text: str = Field(
        ..., description="Narrative diagnosis in plain clinical language"
    )
    findings: list[Finding] = Field(
        ...,
        description="Structured evidence trail — one Finding per piece of evidence",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence score (0.0–1.0). Used by the GP Orchestrator "
            "for weighting and by the Consensus Auditor for trustworthiness scoring"
        ),
    )
    reasoning: str = Field(
        ...,
        description="The specialist's clinical reasoning narrative (LLM-generated)",
    )
    data_quality_flag: str = Field(
        default="adequate",
        description=(
            "Assessment of input data quality: 'adequate', 'limited', or "
            "'insufficient'. 'insufficient' signals the opinion should be "
            "down-weighted or excluded from consensus"
        ),
    )
