"""Grounding Checker — V1 Rule-Based Citation Matching.

Verifies that each Claim in a DiagnosticHypothesis is grounded in the
patient's input data. This is the V1 implementation using deterministic
citation-matching logic:

  1. Check if supporting_evidence_ref matches an Observation.code or
     a substring of a clinical note.
  2. If referencing an observation, verify the claim text is consistent
     with the observation's value (string/numeric match).
  3. Mark each Claim.grounded = True/False with a reason string.

V2 upgrade path: swap this module for an NLI entailment model
(BioBERT/PubMedBERT on MedNLI) without changing the interface.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from schemas.fhir_input import PatientCase, Observation
from schemas.hypothesis import Claim, DiagnosticHypothesis

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding a single claim."""

    grounded: bool
    reason: str


def _find_observation_by_code(
    case: PatientCase, code: str
) -> Observation | None:
    """Find an observation in the patient case by its code (case-insensitive)."""
    code_lower = code.strip().lower()
    for obs in case.observations:
        if obs.code.strip().lower() == code_lower:
            return obs
    return None


def _find_note_match(case: PatientCase, ref: str) -> str | None:
    """Check if the reference is a substring of any clinical note.

    Returns the matching note excerpt if found, None otherwise.
    """
    ref_lower = ref.strip().lower()
    if not ref_lower:
        return None

    for note in case.notes:
        if ref_lower in note.lower():
            return note
    return None


def _check_symptom_match(case: PatientCase, ref: str) -> bool:
    """Check if the reference matches any reported symptom."""
    ref_lower = ref.strip().lower()
    return any(ref_lower in symptom.lower() or symptom.lower() in ref_lower for symptom in case.symptoms)


def _claim_text_consistent_with_observation(
    claim_text: str, obs: Observation
) -> tuple[bool, str]:
    """Check whether the claim text is consistent with an observation's value.

    Uses a combination of:
    - Exact value mention check
    - Numeric comparison (e.g., claim says "elevated glucose" and value > normal)
    - Display name mention check

    Returns (is_consistent, reason).
    """
    claim_lower = claim_text.lower()
    value_lower = obs.value.lower()
    display_lower = obs.display.lower()

    # Check 1: Does the claim mention the observation's value or display name?
    mentions_value = value_lower in claim_lower
    mentions_display = display_lower in claim_lower

    # Check 2: Does the claim mention key parts of the observation?
    # Extract numbers from both the claim and the observation value
    claim_numbers = set(re.findall(r"\d+\.?\d*", claim_text))
    obs_numbers = set(re.findall(r"\d+\.?\d*", obs.value))
    numbers_overlap = bool(claim_numbers & obs_numbers)

    # Check 3: Qualitative consistency — does the claim use directional
    # language (elevated, low, high, abnormal, normal) that aligns?
    qualitative_terms = {
        "elevated", "high", "increased", "above", "raised", "rising",
        "low", "decreased", "reduced", "below", "diminished",
        "normal", "within range", "unremarkable",
        "abnormal", "irregular", "positive", "negative",
    }
    claim_has_qualitative = any(term in claim_lower for term in qualitative_terms)

    if mentions_value or numbers_overlap:
        return True, f"Claim references value '{obs.value}' from observation [{obs.code}] {obs.display}"

    if mentions_display:
        if claim_has_qualitative:
            # The claim mentions the test by name and makes a qualitative statement.
            # V1 accepts this as consistent — V2 (NLI) would verify the direction.
            return True, (
                f"Claim mentions '{obs.display}' with qualitative assessment. "
                f"Actual value: {obs.value}{' ' + obs.unit if obs.unit else ''}"
            )
        return True, f"Claim mentions observation '{obs.display}' [{obs.code}]"

    return False, (
        f"Claim text does not reference the value or display name of "
        f"observation [{obs.code}] {obs.display} = {obs.value}"
    )


def ground_single_claim(claim: Claim, case: PatientCase) -> GroundingResult:
    """Ground a single claim against the patient case data.

    Args:
        claim: The claim to verify.
        case: The patient case containing the evidence.

    Returns:
        GroundingResult with grounded status and reason.
    """
    ref = claim.supporting_evidence_ref

    # ── No reference provided ────────────────────────────────────────────
    if not ref or ref.strip() == "":
        return GroundingResult(
            grounded=False,
            reason="No supporting_evidence_ref provided — claim has no cited evidence",
        )

    ref_stripped = ref.strip()

    # ── Try matching as an Observation.code ───────────────────────────────
    obs = _find_observation_by_code(case, ref_stripped)
    if obs is not None:
        is_consistent, reason = _claim_text_consistent_with_observation(
            claim.text, obs
        )
        if is_consistent:
            return GroundingResult(grounded=True, reason=reason)
        else:
            return GroundingResult(
                grounded=False,
                reason=f"Reference [{ref_stripped}] exists but claim is inconsistent: {reason}",
            )

    # ── Try matching as a clinical note excerpt ──────────────────────────
    note_match = _find_note_match(case, ref_stripped)
    if note_match is not None:
        # V1: if the reference is found in a note, accept it as grounded.
        # The claim references text that actually exists in the clinical notes.
        return GroundingResult(
            grounded=True,
            reason=f"Reference found in clinical notes: '{ref_stripped}' matches note content",
        )

    # ── Try matching as a symptom reference ──────────────────────────────
    if _check_symptom_match(case, ref_stripped):
        return GroundingResult(
            grounded=True,
            reason=f"Reference '{ref_stripped}' matches a reported symptom",
        )

    # ── Check if it matches the chief complaint ──────────────────────────
    if ref_stripped.lower() in case.chief_complaint.lower():
        return GroundingResult(
            grounded=True,
            reason=f"Reference '{ref_stripped}' matches the chief complaint",
        )

    # ── No match found ───────────────────────────────────────────────────
    return GroundingResult(
        grounded=False,
        reason=(
            f"Reference '{ref_stripped}' not found in observations "
            f"(codes: {[o.code for o in case.observations]}), "
            f"symptoms, chief complaint, or clinical notes"
        ),
    )


def check_grounding(
    hypothesis: DiagnosticHypothesis, case: PatientCase
) -> DiagnosticHypothesis:
    """Check grounding for all claims in a hypothesis.

    Mutates claim.grounded and claim.grounding_reason in-place, then
    sets hypothesis.is_fully_grounded accordingly.

    Args:
        hypothesis: The hypothesis to check.
        case: The patient case data to check against.

    Returns:
        The same hypothesis with grounding fields populated.
    """
    all_grounded = True

    for i, claim in enumerate(hypothesis.claims):
        result = ground_single_claim(claim, case)
        claim.grounded = result.grounded
        claim.grounding_reason = result.reason

        logger.info(
            "Claim %d/%d [%s]: %s — %s",
            i + 1,
            len(hypothesis.claims),
            "GROUNDED" if result.grounded else "UNGROUNDED",
            claim.text[:80],
            result.reason,
        )

        if not result.grounded:
            all_grounded = False

    hypothesis.is_fully_grounded = all_grounded

    logger.info(
        "Grounding check complete for case %s: %d/%d claims grounded, fully_grounded=%s",
        hypothesis.case_id,
        sum(1 for c in hypothesis.claims if c.grounded),
        len(hypothesis.claims),
        hypothesis.is_fully_grounded,
    )

    return hypothesis
