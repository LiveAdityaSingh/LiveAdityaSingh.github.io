"""Base specialist interface — abstract contract all specialists implement.

Enforces the frozen-state constraint: specialists receive only raw patient
data + domain-specific inputs, never each other's outputs or the
orchestrator's reasoning. This architectural boundary prevents bias cascade
(groupthink) across the specialist agents.

All specialist agents in agents/specialists/ must subclass BaseSpecialist
and implement the analyze() method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from schemas.fhir_input import PatientCase
from schemas.specialist_opinion import SpecialistOpinion

logger = logging.getLogger(__name__)


class BaseSpecialist(ABC):
    """Abstract base class for all specialist agents.

    Subclasses must:
    1. Set `specialty` to their domain name (e.g. "cardiology").
    2. Implement `analyze()` to produce a SpecialistOpinion from raw data.
    3. Never access other specialists' outputs (enforced by the LangGraph
       state schema in M7, but respected by convention from M2 onward).
    """

    specialty: str = "unspecified"

    @abstractmethod
    def analyze(
        self, case: PatientCase, **domain_data
    ) -> SpecialistOpinion:
        """Produce a specialist opinion from raw patient data.

        Args:
            case: The patient case (demographics, symptoms, observations,
                  clinical notes). This is the ONLY shared input across
                  all specialists.
            **domain_data: Specialist-specific inputs. Examples:
                - Cardiology: ecg_waveform (np.ndarray), sampling_rate (int)
                - Neurology: imaging_report (str)
                - Surgery: procedure_notes (str)
                These are raw data only — never another specialist's output.

        Returns:
            A validated SpecialistOpinion.

        Raises:
            ValueError: If required domain_data is missing.
            RuntimeError: If the analysis pipeline fails unrecoverably.
        """
        ...

    def _validate_case(self, case: PatientCase) -> None:
        """Basic validation that the case has minimum required data."""
        if not case.case_id:
            raise ValueError("PatientCase must have a case_id")
        if not case.symptoms and not case.observations and not case.notes:
            logger.warning(
                "Case %s has no symptoms, observations, or notes — "
                "specialist analysis may produce low-quality output",
                case.case_id,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} specialty='{self.specialty}'>"
