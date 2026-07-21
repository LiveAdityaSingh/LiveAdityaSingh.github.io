"""Unit tests for Milestone 1 — Primary Diagnostic LLM + Grounding Loop.

Tests cover:
  - Schema validation (valid/invalid PatientCase and DiagnosticHypothesis)
  - Grounding checker logic (grounded claims pass, ungrounded claims fail)
  - Loop termination (always ≤3 iterations)
  - Malformed LLM output handling (never crashes)

All LLM calls are mocked — these tests run without Ollama or a model.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from schemas.fhir_input import PatientCase, Observation
from schemas.hypothesis import Claim, DiagnosticHypothesis
from agents.primary_llm.grounding_checker import (
    check_grounding,
    ground_single_claim,
)
from agents.primary_llm.loop import run_grounding_loop


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_case() -> PatientCase:
    """A minimal patient case for testing."""
    return PatientCase(
        case_id="TEST-001",
        age=58,
        sex="M",
        chief_complaint="Chest pain radiating to left arm",
        symptoms=["chest pain", "diaphoresis", "shortness of breath"],
        observations=[
            Observation(code="6598-7", display="Troponin T", value="0.85", unit="ng/mL"),
            Observation(code="2157-6", display="Creatine kinase-MB", value="45", unit="ng/mL"),
            Observation(code="8867-4", display="Heart rate", value="110", unit="bpm"),
        ],
        notes=["Patient has a 30-year smoking history. Previous angioplasty in 2019."],
    )


@pytest.fixture
def grounded_hypothesis(sample_case: PatientCase) -> DiagnosticHypothesis:
    """A hypothesis with all claims properly grounded."""
    return DiagnosticHypothesis(
        case_id=sample_case.case_id,
        hypothesis_text="Suspected acute myocardial infarction",
        claims=[
            Claim(text="Troponin T is elevated at 0.85 ng/mL", supporting_evidence_ref="6598-7"),
            Claim(text="CK-MB is elevated at 45 ng/mL", supporting_evidence_ref="2157-6"),
            Claim(text="Patient has a 30-year smoking history", supporting_evidence_ref="30-year smoking history"),
        ],
    )


@pytest.fixture
def ungrounded_hypothesis(sample_case: PatientCase) -> DiagnosticHypothesis:
    """A hypothesis with fabricated claims."""
    return DiagnosticHypothesis(
        case_id=sample_case.case_id,
        hypothesis_text="Suspected pulmonary embolism",
        claims=[
            Claim(text="D-dimer is elevated at 2.5", supporting_evidence_ref="48065-7"),
            Claim(text="CT pulmonary angiogram shows filling defect", supporting_evidence_ref="CT findings"),
            Claim(text="Patient has history of DVT", supporting_evidence_ref=None),
        ],
    )


# ── Schema Validation Tests ──────────────────────────────────────────────────

class TestSchemaValidation:
    """Test Pydantic schema validation for input and output models."""

    def test_valid_patient_case(self, sample_case: PatientCase):
        """Valid PatientCase should parse without errors."""
        assert sample_case.case_id == "TEST-001"
        assert len(sample_case.observations) == 3
        assert sample_case.age == 58

    def test_patient_case_invalid_age(self):
        """Age out of range should raise ValidationError."""
        with pytest.raises(ValidationError):
            PatientCase(
                case_id="BAD-001",
                age=-5,
                sex="M",
                chief_complaint="Test",
                symptoms=["pain"],
                observations=[],
            )

    def test_patient_case_missing_required_fields(self):
        """Missing required fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            PatientCase(case_id="BAD-002", age=30)  # type: ignore

    def test_valid_hypothesis(self, grounded_hypothesis: DiagnosticHypothesis):
        """Valid DiagnosticHypothesis should parse without errors."""
        assert grounded_hypothesis.case_id == "TEST-001"
        assert len(grounded_hypothesis.claims) == 3
        assert grounded_hypothesis.is_fully_grounded is False  # default

    def test_observation_optional_fields(self):
        """Observation with only required fields should be valid."""
        obs = Observation(code="12345", display="Test obs", value="100")
        assert obs.unit is None
        assert obs.effective_time is None

    def test_claim_defaults(self):
        """Claim should have sensible defaults for grounding fields."""
        claim = Claim(text="Test claim")
        assert claim.grounded is None
        assert claim.grounding_reason is None
        assert claim.supporting_evidence_ref is None


# ── Grounding Checker Tests ──────────────────────────────────────────────────

class TestGroundingChecker:
    """Test the V1 rule-based grounding checker."""

    def test_grounded_claim_by_observation_code(self, sample_case: PatientCase):
        """Claim referencing a valid observation code with consistent text should be grounded."""
        claim = Claim(
            text="Troponin T is elevated at 0.85 ng/mL",
            supporting_evidence_ref="6598-7",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is True

    def test_ungrounded_claim_nonexistent_code(self, sample_case: PatientCase):
        """Claim referencing a non-existent observation code should be ungrounded."""
        claim = Claim(
            text="D-dimer is elevated at 2.5",
            supporting_evidence_ref="48065-7",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is False

    def test_ungrounded_claim_no_reference(self, sample_case: PatientCase):
        """Claim with no supporting_evidence_ref should be ungrounded."""
        claim = Claim(
            text="Patient should receive aspirin",
            supporting_evidence_ref=None,
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is False

    def test_ungrounded_claim_empty_reference(self, sample_case: PatientCase):
        """Claim with empty string reference should be ungrounded."""
        claim = Claim(
            text="Patient has diabetes",
            supporting_evidence_ref="",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is False

    def test_grounded_claim_by_note_match(self, sample_case: PatientCase):
        """Claim referencing text found in clinical notes should be grounded."""
        claim = Claim(
            text="Patient has a 30-year smoking history",
            supporting_evidence_ref="30-year smoking history",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is True

    def test_ungrounded_claim_note_not_found(self, sample_case: PatientCase):
        """Claim referencing text NOT in clinical notes should be ungrounded."""
        claim = Claim(
            text="Patient has history of DVT",
            supporting_evidence_ref="history of DVT",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is False

    def test_grounded_claim_by_symptom(self, sample_case: PatientCase):
        """Claim referencing a reported symptom should be grounded."""
        claim = Claim(
            text="Patient presents with chest pain",
            supporting_evidence_ref="chest pain",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is True

    def test_grounded_claim_by_chief_complaint(self, sample_case: PatientCase):
        """Claim referencing the chief complaint should be grounded."""
        claim = Claim(
            text="Patient complains of chest pain radiating to left arm",
            supporting_evidence_ref="Chest pain radiating to left arm",
        )
        result = ground_single_claim(claim, sample_case)
        assert result.grounded is True

    def test_check_grounding_all_grounded(
        self, sample_case: PatientCase, grounded_hypothesis: DiagnosticHypothesis
    ):
        """Hypothesis with all grounded claims should have is_fully_grounded=True."""
        result = check_grounding(grounded_hypothesis, sample_case)
        assert result.is_fully_grounded is True
        assert all(c.grounded for c in result.claims)

    def test_check_grounding_some_ungrounded(
        self, sample_case: PatientCase, ungrounded_hypothesis: DiagnosticHypothesis
    ):
        """Hypothesis with ungrounded claims should have is_fully_grounded=False."""
        result = check_grounding(ungrounded_hypothesis, sample_case)
        assert result.is_fully_grounded is False
        assert not all(c.grounded for c in result.claims)

    def test_grounding_reason_always_set(
        self, sample_case: PatientCase, grounded_hypothesis: DiagnosticHypothesis
    ):
        """Every claim should have a grounding_reason after checking."""
        result = check_grounding(grounded_hypothesis, sample_case)
        for claim in result.claims:
            assert claim.grounding_reason is not None
            assert len(claim.grounding_reason) > 0


# ── Grounding Loop Tests (Mocked LLM) ────────────────────────────────────────

class TestGroundingLoop:
    """Test the generate → check → re-prompt loop with mocked LLM."""

    def _mock_ollama_response(self, hypothesis_dict: dict) -> dict:
        """Create a mock Ollama response from a hypothesis dict."""
        return {
            "message": {
                "content": json.dumps(hypothesis_dict),
            }
        }

    def test_loop_terminates_on_grounded(self, sample_case: PatientCase):
        """Loop should stop after 1 iteration if hypothesis is fully grounded."""
        mock_response = self._mock_ollama_response({
            "case_id": "TEST-001",
            "hypothesis_text": "Acute MI",
            "claims": [
                {"text": "Troponin T is elevated at 0.85 ng/mL", "supporting_evidence_ref": "6598-7"},
                {"text": "Heart rate is 110 bpm indicating tachycardia", "supporting_evidence_ref": "8867-4"},
            ],
        })

        with patch("agents.primary_llm.generate.ollama.chat", return_value=mock_response):
            hypothesis, audit_log = run_grounding_loop(sample_case)

        assert hypothesis.is_fully_grounded is True
        assert len(audit_log) == 1  # only 1 iteration needed
        assert audit_log[0]["iteration"] == 1

    def test_loop_hard_cap_at_3_iterations(self, sample_case: PatientCase):
        """Loop should NEVER exceed 3 iterations, even if never grounded."""
        # Always return ungrounded claims
        mock_response = self._mock_ollama_response({
            "case_id": "TEST-001",
            "hypothesis_text": "Fabricated hypothesis",
            "claims": [
                {"text": "D-dimer is elevated", "supporting_evidence_ref": "48065-7"},
            ],
        })

        with patch("agents.primary_llm.generate.ollama.chat", return_value=mock_response):
            hypothesis, audit_log = run_grounding_loop(sample_case, max_iterations=3)

        assert len(audit_log) == 3
        assert hypothesis.is_fully_grounded is False

    def test_loop_never_silently_passes_ungrounded(self, sample_case: PatientCase):
        """An ungrounded hypothesis must NEVER have is_fully_grounded=True."""
        mock_response = self._mock_ollama_response({
            "case_id": "TEST-001",
            "hypothesis_text": "Bad hypothesis",
            "claims": [
                {"text": "Fabricated claim", "supporting_evidence_ref": "FAKE-CODE"},
            ],
        })

        with patch("agents.primary_llm.generate.ollama.chat", return_value=mock_response):
            hypothesis, audit_log = run_grounding_loop(sample_case)

        assert hypothesis.is_fully_grounded is False

    def test_loop_handles_generation_failure(self, sample_case: PatientCase):
        """Loop should handle LLM failures gracefully, not crash."""
        with patch(
            "agents.primary_llm.generate.ollama.chat",
            side_effect=Exception("Connection refused"),
        ):
            hypothesis, audit_log = run_grounding_loop(sample_case)

        assert hypothesis.is_fully_grounded is False
        assert "[GENERATION FAILED" in hypothesis.hypothesis_text

    def test_loop_audit_log_structure(self, sample_case: PatientCase):
        """Audit log entries must contain required fields."""
        mock_response = self._mock_ollama_response({
            "case_id": "TEST-001",
            "hypothesis_text": "Test",
            "claims": [
                {"text": "Troponin T is elevated at 0.85", "supporting_evidence_ref": "6598-7"},
            ],
        })

        with patch("agents.primary_llm.generate.ollama.chat", return_value=mock_response):
            _, audit_log = run_grounding_loop(sample_case)

        assert len(audit_log) >= 1
        entry = audit_log[0]
        assert "case_id" in entry
        assert "timestamp" in entry
        assert "iteration" in entry
        assert "claims" in entry
        assert "is_fully_grounded" in entry
        assert "grounded_count" in entry
        assert "total_claims" in entry
        assert entry["case_id"] == "TEST-001"

    def test_loop_respects_custom_max_iterations(self, sample_case: PatientCase):
        """Loop should respect a custom max_iterations parameter."""
        mock_response = self._mock_ollama_response({
            "case_id": "TEST-001",
            "hypothesis_text": "Bad hypothesis",
            "claims": [
                {"text": "Fabricated", "supporting_evidence_ref": "FAKE"},
            ],
        })

        with patch("agents.primary_llm.generate.ollama.chat", return_value=mock_response):
            _, audit_log = run_grounding_loop(sample_case, max_iterations=1)

        assert len(audit_log) == 1


# ── Malformed Output Tests ───────────────────────────────────────────────────

class TestMalformedOutput:
    """Test that malformed LLM output never crashes the pipeline."""

    def test_invalid_json_triggers_retry(self, sample_case: PatientCase):
        """Invalid JSON from LLM should trigger retry, not crash."""
        from agents.primary_llm.generate import generate_hypothesis

        call_count = 0

        def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"message": {"content": "not valid json {"}}
            return {"message": {"content": json.dumps({
                "case_id": "TEST-001",
                "hypothesis_text": "Valid on retry",
                "claims": [{"text": "Troponin T is elevated", "supporting_evidence_ref": "6598-7"}],
            })}}

        with patch("agents.primary_llm.generate.ollama.chat", side_effect=mock_chat):
            result = generate_hypothesis(sample_case)

        assert result.hypothesis_text == "Valid on retry"
        assert call_count == 3  # 2 failures + 1 success

    def test_exhausted_retries_raises(self, sample_case: PatientCase):
        """Exhausting all retries should raise RuntimeError, not hang."""
        from agents.primary_llm.generate import generate_hypothesis

        with patch(
            "agents.primary_llm.generate.ollama.chat",
            return_value={"message": {"content": "always broken {"}},
        ):
            with pytest.raises(RuntimeError, match="Failed to generate"):
                generate_hypothesis(sample_case)
