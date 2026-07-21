"""Diagnostic hypothesis schema — output of the Primary LLM Agent.

Defines the structured contract for diagnostic hypotheses, including
decomposed claims that can be individually grounded-checked.
All downstream milestones depend on these exact field names.
"""

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """An atomic, checkable statement extracted from a diagnostic hypothesis.

    Each claim references a specific piece of evidence from the patient case
    and can be independently verified by the grounding checker.
    """

    text: str = Field(
        ..., description="The atomic claim statement"
    )
    supporting_evidence_ref: str | None = Field(
        default=None,
        description=(
            "Reference to supporting evidence: an Observation.code "
            "or a verbatim excerpt from a clinical note"
        ),
    )
    grounded: bool | None = Field(
        default=None,
        description="Set by grounding_checker: True if claim is supported by evidence",
    )
    grounding_reason: str | None = Field(
        default=None,
        description="Explanation of why the claim is grounded or ungrounded (audit trail)",
    )


class DiagnosticHypothesis(BaseModel):
    """A structured diagnostic hypothesis with decomposed, checkable claims.

    Produced by the Primary LLM Agent and validated by the grounding loop.
    """

    case_id: str = Field(
        ..., description="The patient case this hypothesis refers to"
    )
    hypothesis_text: str = Field(
        ..., description="The full diagnostic hypothesis as natural language"
    )
    claims: list[Claim] = Field(
        ..., description="Decomposed atomic claims that support the hypothesis"
    )
    iteration: int = Field(
        default=0, description="Which generation iteration produced this hypothesis"
    )
    is_fully_grounded: bool = Field(
        default=False,
        description="True only if ALL claims are grounded; never set True if any claim is ungrounded",
    )
