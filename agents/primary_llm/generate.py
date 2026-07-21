"""Primary Diagnostic LLM — Hypothesis Generator.

Calls OpenBioLLM-Llama3-8B via Ollama to produce a structured
DiagnosticHypothesis from a PatientCase. Enforces structured JSON
output using Ollama's native constrained decoding.

This module is the first step in the grounding loop:
  generate → grounding_check → (re-prompt if needed) → repeat
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import ollama
from pydantic import ValidationError

from schemas.fhir_input import PatientCase
from schemas.hypothesis import Claim, DiagnosticHypothesis

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openbiollm"
MAX_SCHEMA_RETRIES = 2  # retries on malformed JSON, not grounding retries

SYSTEM_PROMPT = """\
You are a clinical decision-support assistant. Your role is to generate a
diagnostic hypothesis for a patient case and decompose it into atomic,
verifiable claims.

CRITICAL CONSTRAINTS:
- You are a DECISION-SUPPORT tool, NOT an autonomous diagnostic system.
  A human clinician will always review your output.
- Every claim MUST reference specific evidence from the patient input.
- Do NOT fabricate evidence. If you cannot ground a claim, say so explicitly.
- Do NOT hallucinate lab values, symptoms, or observations not present in the input.

OUTPUT FORMAT:
You must respond with a JSON object matching this exact structure:
{
  "case_id": "<same as input case_id>",
  "hypothesis_text": "<your diagnostic hypothesis as a single paragraph>",
  "claims": [
    {
      "text": "<an atomic, verifiable statement>",
      "supporting_evidence_ref": "<the Observation.code OR a verbatim note excerpt that supports this claim>"
    }
  ]
}

RULES FOR CLAIMS:
1. Each claim must be a single, atomic statement (one fact per claim).
2. supporting_evidence_ref must point to either:
   - An Observation.code from the patient's observations (e.g. "2093-3"), OR
   - A verbatim excerpt from the patient's clinical notes.
3. Decompose your hypothesis into 3–8 claims. Fewer if the case is simple.
4. Do not include claims that cannot be traced to the input data.
"""


def _build_user_prompt(
    case: PatientCase, feedback: str | None = None
) -> str:
    """Build the user prompt from a PatientCase, optionally including
    feedback from a failed grounding check."""

    observations_block = "\n".join(
        f"  - [{obs.code}] {obs.display}: {obs.value}"
        + (f" {obs.unit}" if obs.unit else "")
        for obs in case.observations
    )
    notes_block = "\n".join(f"  - {note}" for note in case.notes) if case.notes else "  (none)"
    symptoms_block = "\n".join(f"  - {s}" for s in case.symptoms)

    prompt = f"""\
PATIENT CASE (ID: {case.case_id})
- Age: {case.age}
- Sex: {case.sex}
- Chief complaint: {case.chief_complaint}

Symptoms:
{symptoms_block}

Observations (labs/vitals):
{observations_block}

Clinical notes:
{notes_block}

Generate a diagnostic hypothesis with grounded claims for this patient case.\
"""

    if feedback:
        prompt += f"""

IMPORTANT — PREVIOUS ATTEMPT FAILED GROUNDING CHECK:
{feedback}

Please regenerate your hypothesis, fixing the ungrounded claims listed above.
Only include claims you can directly support with evidence from the patient data.\
"""

    return prompt


def _get_hypothesis_json_schema() -> dict[str, Any]:
    """Return the JSON schema that Ollama will use for constrained decoding.

    This schema enforces the structure at the decoding level, so the model
    is physically unable to produce output that doesn't match.
    """
    return {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "hypothesis_text": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "supporting_evidence_ref": {"type": "string"},
                    },
                    "required": ["text", "supporting_evidence_ref"],
                },
            },
        },
        "required": ["case_id", "hypothesis_text", "claims"],
    }


def generate_hypothesis(
    case: PatientCase,
    feedback: str | None = None,
    iteration: int = 0,
    model: str | None = None,
) -> DiagnosticHypothesis:
    """Generate a diagnostic hypothesis for a patient case.

    Args:
        case: The patient case to diagnose.
        feedback: Optional feedback from a failed grounding check (for re-prompts).
        iteration: Current loop iteration number.
        model: Ollama model name override (defaults to OLLAMA_MODEL env var).

    Returns:
        A validated DiagnosticHypothesis.

    Raises:
        RuntimeError: If the model fails to produce valid output after retries.
    """
    model_name = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    user_prompt = _build_user_prompt(case, feedback)

    last_error: Exception | None = None

    for attempt in range(1, MAX_SCHEMA_RETRIES + 2):  # +2 because range is exclusive + 1 initial
        logger.info(
            "Generating hypothesis for case %s (iteration=%d, schema_attempt=%d, model=%s)",
            case.case_id,
            iteration,
            attempt,
            model_name,
        )

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=_get_hypothesis_json_schema(),
            )

            raw_content = response["message"]["content"]
            logger.debug("Raw LLM output: %s", raw_content)

            # Parse JSON and validate against Pydantic schema
            parsed = json.loads(raw_content)

            # Build Claim objects (LLM won't set grounded/grounding_reason)
            claims = [
                Claim(
                    text=c["text"],
                    supporting_evidence_ref=c.get("supporting_evidence_ref"),
                )
                for c in parsed.get("claims", [])
            ]

            hypothesis = DiagnosticHypothesis(
                case_id=case.case_id,  # always use the real case_id, not the LLM's
                hypothesis_text=parsed["hypothesis_text"],
                claims=claims,
                iteration=iteration,
                is_fully_grounded=False,  # set by grounding_checker, never by the LLM
            )

            logger.info(
                "Generated hypothesis with %d claims for case %s",
                len(hypothesis.claims),
                case.case_id,
            )
            return hypothesis

        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Schema validation failed on attempt %d for case %s: %s",
                attempt,
                case.case_id,
                exc,
            )
            continue

        except ollama.ResponseError as exc:
            last_error = exc
            logger.error(
                "Ollama API error on attempt %d for case %s: %s",
                attempt,
                case.case_id,
                exc,
            )
            continue

        except Exception as exc:
            # Catch-all for unexpected errors (connection refused, timeouts, etc.)
            # so the pipeline never crashes on a single LLM call failure.
            last_error = exc
            logger.error(
                "Unexpected error on attempt %d for case %s: %s: %s",
                attempt,
                case.case_id,
                type(exc).__name__,
                exc,
            )
            continue

    # All retries exhausted
    raise RuntimeError(
        f"Failed to generate valid hypothesis for case {case.case_id} "
        f"after {MAX_SCHEMA_RETRIES + 1} attempts. Last error: {last_error}"
    )
