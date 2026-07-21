"""Grounding Loop — Orchestrates generate → check → re-prompt.

This is the main entry point for Milestone 1. It coordinates the
hypothesis generator and grounding checker in a bounded retry loop:

  1. Generate a diagnostic hypothesis from a PatientCase.
  2. Check grounding of every claim.
  3. If any claim is ungrounded, build feedback explaining which claims
     failed and why, then re-prompt the LLM.
  4. Hard cap: 3 iterations. On cap, return the hypothesis with
     is_fully_grounded = False — never silently pass an ungrounded
     hypothesis as if it were validated.

Every iteration is logged for the audit trail (required from M1 onward).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from schemas.fhir_input import PatientCase
from schemas.hypothesis import DiagnosticHypothesis
from agents.primary_llm.generate import generate_hypothesis
from agents.primary_llm.grounding_checker import check_grounding

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MAX_ITERATIONS = 3


def _build_grounding_feedback(hypothesis: DiagnosticHypothesis) -> str:
    """Build human-readable feedback for ungrounded claims.

    This feedback is included in the re-prompt so the LLM knows exactly
    which claims failed and why, enabling targeted correction.
    """
    failed_claims = [c for c in hypothesis.claims if not c.grounded]

    if not failed_claims:
        return ""

    lines = [f"{len(failed_claims)} claim(s) failed grounding check:\n"]
    for i, claim in enumerate(failed_claims, 1):
        lines.append(f"  {i}. UNGROUNDED CLAIM: \"{claim.text}\"")
        lines.append(f"     Reference: {claim.supporting_evidence_ref or '(none)'}")
        lines.append(f"     Reason: {claim.grounding_reason or '(no reason given)'}")
        lines.append("")

    lines.append(
        "Fix these claims by either:\n"
        "  - Providing correct evidence references from the patient data, OR\n"
        "  - Removing claims that cannot be supported by the input evidence."
    )

    return "\n".join(lines)


def _log_iteration(
    case_id: str,
    iteration: int,
    hypothesis: DiagnosticHypothesis,
    audit_log: list[dict[str, Any]],
) -> None:
    """Log a single iteration to the audit trail.

    Logs case_id, timestamp, iteration number, all claims with grounded
    status. This structured log is the audit trail required by the spec.
    """
    entry = {
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "hypothesis_text": hypothesis.hypothesis_text,
        "is_fully_grounded": hypothesis.is_fully_grounded,
        "claims": [
            {
                "text": c.text,
                "supporting_evidence_ref": c.supporting_evidence_ref,
                "grounded": c.grounded,
                "grounding_reason": c.grounding_reason,
            }
            for c in hypothesis.claims
        ],
        "grounded_count": sum(1 for c in hypothesis.claims if c.grounded),
        "total_claims": len(hypothesis.claims),
    }

    audit_log.append(entry)

    logger.info(
        "AUDIT | case=%s | iter=%d | grounded=%d/%d | fully_grounded=%s",
        case_id,
        iteration,
        entry["grounded_count"],
        entry["total_claims"],
        hypothesis.is_fully_grounded,
    )


def run_grounding_loop(
    case: PatientCase,
    model: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[DiagnosticHypothesis, list[dict[str, Any]]]:
    """Run the full generate → ground → re-prompt loop.

    Args:
        case: The patient case to generate a hypothesis for.
        model: Optional Ollama model name override.
        max_iterations: Maximum number of attempts (default 3).

    Returns:
        A tuple of (final_hypothesis, audit_log).
        - final_hypothesis.is_fully_grounded indicates success/failure.
        - audit_log contains structured records of every iteration.

    The loop will NEVER:
    - Run more than max_iterations times.
    - Silently pass an ungrounded hypothesis as fully grounded.
    - Crash on malformed LLM output (handled by generate.py retries).
    """
    audit_log: list[dict[str, Any]] = []
    feedback: str | None = None

    logger.info(
        "Starting grounding loop for case %s (max_iterations=%d)",
        case.case_id,
        max_iterations,
    )

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "── Iteration %d/%d for case %s ──",
            iteration,
            max_iterations,
            case.case_id,
        )

        # Step 1: Generate hypothesis
        try:
            hypothesis = generate_hypothesis(
                case=case,
                feedback=feedback,
                iteration=iteration,
                model=model,
            )
        except RuntimeError as exc:
            # generate.py exhausted its own retries for schema validation.
            # Log the failure and return a placeholder ungrounded hypothesis.
            logger.error(
                "Generation failed for case %s on iteration %d: %s",
                case.case_id,
                iteration,
                exc,
            )
            failed_hypothesis = DiagnosticHypothesis(
                case_id=case.case_id,
                hypothesis_text=f"[GENERATION FAILED: {exc}]",
                claims=[],
                iteration=iteration,
                is_fully_grounded=False,
            )
            _log_iteration(case.case_id, iteration, failed_hypothesis, audit_log)
            return failed_hypothesis, audit_log

        # Step 2: Check grounding
        hypothesis = check_grounding(hypothesis, case)

        # Step 3: Log this iteration
        _log_iteration(case.case_id, iteration, hypothesis, audit_log)

        # Step 4: Check if fully grounded
        if hypothesis.is_fully_grounded:
            logger.info(
                "✓ Case %s fully grounded after %d iteration(s)",
                case.case_id,
                iteration,
            )
            return hypothesis, audit_log

        # Step 5: Build feedback for re-prompt (if more iterations remain)
        if iteration < max_iterations:
            feedback = _build_grounding_feedback(hypothesis)
            logger.info(
                "Case %s not fully grounded, will re-prompt (iteration %d → %d)",
                case.case_id,
                iteration,
                iteration + 1,
            )
        else:
            logger.warning(
                "✗ Case %s exhausted %d iterations without full grounding. "
                "Returning with is_fully_grounded=False.",
                case.case_id,
                max_iterations,
            )

    # Return the last hypothesis — is_fully_grounded is already False
    return hypothesis, audit_log
