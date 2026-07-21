"""FHIR R4 / UK Core input schema for patient case data.

These Pydantic models define the contract for patient input throughout the
MLTVA pipeline. All downstream milestones depend on these exact field names.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    """A single clinical observation (lab result, vital sign, etc.).

    Mirrors FHIR R4 Observation resource with the fields relevant to
    diagnostic hypothesis generation.
    """

    code: str = Field(
        ..., description="LOINC or local code identifying this observation"
    )
    display: str = Field(
        ..., description="Human-readable name of the observation"
    )
    value: str = Field(
        ..., description="Observed value (numeric or categorical as string)"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measurement (e.g. mg/dL, mmHg)"
    )
    effective_time: Optional[str] = Field(
        default=None, description="ISO 8601 datetime when observation was taken"
    )


class PatientCase(BaseModel):
    """Structured patient case input for the diagnostic pipeline.

    Represents a de-identified patient encounter with symptoms, observations,
    and clinical notes. This is the entry point for the entire MLTVA system.
    """

    case_id: str = Field(
        ..., description="Unique identifier for this patient case"
    )
    age: int = Field(
        ..., ge=0, le=150, description="Patient age in years"
    )
    sex: str = Field(
        ..., description="Patient sex (M/F/Other)"
    )
    chief_complaint: str = Field(
        ..., description="Primary reason for the clinical encounter"
    )
    symptoms: list[str] = Field(
        ..., description="List of reported symptoms"
    )
    observations: list[Observation] = Field(
        ..., description="Lab results, vital signs, and other clinical observations"
    )
    notes: list[str] = Field(
        default_factory=list, description="Free-text clinical notes"
    )
