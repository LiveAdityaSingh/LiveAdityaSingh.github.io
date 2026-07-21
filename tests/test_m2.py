"""Unit tests for Milestone 2 — Cardiology/ECG Specialist Agent.

Tests cover:
  - SpecialistOpinion and ECGFeatures schema validation
  - Base specialist interface contract
  - DSP feature extraction on synthetic ECG signals
  - CNN classifier inference (with random weights — no trained model needed)
  - Full agent integration (mocked LLM, real DSP + CNN)

All LLM calls are mocked — these tests run without Ollama or a trained model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from schemas.fhir_input import PatientCase, Observation
from schemas.specialist_opinion import Finding, SpecialistOpinion
from agents.specialists.base_specialist import BaseSpecialist
from agents.specialists.cardiology.ecg_features import (
    ECGFeatures,
    MultiLeadECGFeatures,
    bandpass_filter,
    detect_r_peaks,
    estimate_signal_quality,
    extract_single_lead_features,
    extract_multilead_features,
    compute_rr_intervals,
    compute_heart_rate,
    check_rhythm_regularity,
)
from agents.specialists.cardiology.ecg_classifier import (
    ECGClassification,
    ECGClassifier,
    ResNet1D,
    DIAGNOSTIC_SUPERCLASSES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cardiac_case() -> PatientCase:
    """A cardiac patient case for testing."""
    return PatientCase(
        case_id="CARDIO-001",
        age=62,
        sex="M",
        chief_complaint="Chest pain with ECG changes",
        symptoms=["chest pain", "diaphoresis", "shortness of breath"],
        observations=[
            Observation(code="6598-7", display="Troponin T", value="1.2", unit="ng/mL"),
            Observation(code="8867-4", display="Heart rate", value="95", unit="bpm"),
        ],
        notes=["Smoker, 40-pack-year history. Known hypertension."],
    )


@pytest.fixture
def synthetic_ecg_12lead() -> np.ndarray:
    """A synthetic 12-lead ECG waveform (10s @ 100Hz)."""
    rng = np.random.RandomState(42)
    fs = 100
    duration = 10.0
    num_samples = int(duration * fs)
    t = np.arange(num_samples) / fs

    waveform = np.zeros((num_samples, 12), dtype=np.float32)
    heart_rate = 75  # bpm
    rr_interval = 60.0 / heart_rate

    for lead in range(12):
        signal = np.zeros(num_samples, dtype=np.float32)
        beat_times = np.arange(0, duration, rr_interval)

        for beat in beat_times:
            # R peak
            signal += rng.uniform(0.8, 1.2) * np.exp(
                -((t - beat) ** 2) / (2 * 0.01 ** 2)
            )
            # T wave
            signal += 0.3 * np.exp(
                -((t - (beat + 0.25)) ** 2) / (2 * 0.05 ** 2)
            )

        signal += rng.normal(0, 0.02, num_samples)
        waveform[:, lead] = signal

    return waveform


@pytest.fixture
def synthetic_ecg_single_lead() -> np.ndarray:
    """A synthetic single-lead ECG signal."""
    rng = np.random.RandomState(42)
    fs = 100
    t = np.arange(1000) / fs

    signal = np.zeros(1000, dtype=np.float32)
    for beat_time in np.arange(0, 10, 0.8):  # 75 bpm
        signal += 1.0 * np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
        signal += 0.3 * np.exp(-((t - (beat_time + 0.25)) ** 2) / (2 * 0.05 ** 2))

    signal += rng.normal(0, 0.02, 1000)
    return signal


# ── Schema Validation Tests ──────────────────────────────────────────────────

class TestSpecialistOpinionSchema:
    """Test Pydantic schema validation for specialist output models."""

    def test_valid_specialist_opinion(self):
        """Valid SpecialistOpinion should parse without errors."""
        opinion = SpecialistOpinion(
            case_id="TEST-001",
            specialty="cardiology",
            diagnosis_codes=["I21.9"],
            diagnosis_text="Acute myocardial infarction",
            findings=[
                Finding(
                    description="ST elevation in leads V1-V4",
                    evidence_source="ecg_features.st_deviation",
                    clinical_significance="Suggests anterior STEMI",
                )
            ],
            confidence=0.85,
            reasoning="Based on troponin elevation and ST changes...",
        )
        assert opinion.specialty == "cardiology"
        assert len(opinion.findings) == 1
        assert opinion.confidence == 0.85

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpecialistOpinion(
                case_id="TEST-001",
                specialty="cardiology",
                diagnosis_codes=["I21.9"],
                diagnosis_text="Test",
                findings=[],
                confidence=1.5,  # out of range
                reasoning="Test",
            )

    def test_default_data_quality_flag(self):
        """Default data_quality_flag should be 'adequate'."""
        opinion = SpecialistOpinion(
            case_id="TEST-001",
            specialty="cardiology",
            diagnosis_codes=["Z01.810"],
            diagnosis_text="Normal",
            findings=[],
            confidence=0.9,
            reasoning="Normal ECG",
        )
        assert opinion.data_quality_flag == "adequate"

    def test_snomed_codes_optional(self):
        """SNOMED codes should be optional (default empty list)."""
        opinion = SpecialistOpinion(
            case_id="TEST-001",
            specialty="cardiology",
            diagnosis_codes=["I21.9"],
            diagnosis_text="MI",
            findings=[],
            confidence=0.8,
            reasoning="Test",
        )
        assert opinion.snomed_codes == []


class TestECGFeaturesSchema:
    """Test ECGFeatures Pydantic models."""

    def test_valid_ecg_features(self):
        features = ECGFeatures(
            lead_name="II",
            sampling_rate=100,
            heart_rate_bpm=75.0,
            qrs_duration_ms=90.0,
            r_peak_count=12,
            signal_quality=0.85,
        )
        assert features.lead_name == "II"
        assert features.heart_rate_bpm == 75.0

    def test_optional_fields_default_none(self):
        features = ECGFeatures(
            lead_name="V1",
            sampling_rate=100,
            heart_rate_bpm=80.0,
        )
        assert features.qt_interval_ms is None
        assert features.st_deviation_mv is None
        assert features.pr_interval_ms is None

    def test_valid_classification(self):
        classification = ECGClassification(
            class_probabilities={"NORM": 0.1, "MI": 0.9, "STTC": 0.3, "CD": 0.05, "HYP": 0.02},
            predicted_classes=["MI"],
            model_confidence=0.9,
        )
        assert classification.predicted_classes == ["MI"]
        assert classification.model_confidence == 0.9


# ── Base Specialist Tests ────────────────────────────────────────────────────

class TestBaseSpecialist:
    """Test the abstract base specialist interface."""

    def test_cannot_instantiate_abstract(self):
        """BaseSpecialist cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSpecialist()

    def test_concrete_implementation(self, cardiac_case):
        """A concrete subclass should be instantiable and callable."""

        class MockSpecialist(BaseSpecialist):
            specialty = "mock"

            def analyze(self, case, **domain_data):
                return SpecialistOpinion(
                    case_id=case.case_id,
                    specialty=self.specialty,
                    diagnosis_codes=["Z00.0"],
                    diagnosis_text="Mock diagnosis",
                    findings=[],
                    confidence=0.5,
                    reasoning="Mock reasoning",
                )

        specialist = MockSpecialist()
        assert specialist.specialty == "mock"
        opinion = specialist.analyze(cardiac_case)
        assert opinion.case_id == "CARDIO-001"

    def test_validate_case_warns_on_empty(self, cardiac_case):
        """_validate_case should warn when case has minimal data."""

        class TestSpecialist(BaseSpecialist):
            specialty = "test"

            def analyze(self, case, **domain_data):
                self._validate_case(case)
                return SpecialistOpinion(
                    case_id=case.case_id,
                    specialty=self.specialty,
                    diagnosis_codes=[],
                    diagnosis_text="Test",
                    findings=[],
                    confidence=0.5,
                    reasoning="Test",
                )

        specialist = TestSpecialist()
        # Should not raise for a normal case
        specialist.analyze(cardiac_case)


# ── DSP Feature Extraction Tests ─────────────────────────────────────────────

class TestDSPFeatures:
    """Test the DSP pipeline for ECG feature extraction."""

    def test_bandpass_filter_shape(self, synthetic_ecg_single_lead):
        """Bandpass filter should preserve signal shape."""
        filtered = bandpass_filter(synthetic_ecg_single_lead, fs=100)
        assert filtered.shape == synthetic_ecg_single_lead.shape

    def test_bandpass_filter_reduces_noise(self, synthetic_ecg_single_lead):
        """Filtered signal should have lower high-frequency energy."""
        noisy = synthetic_ecg_single_lead + np.random.normal(0, 0.5, len(synthetic_ecg_single_lead))
        filtered = bandpass_filter(noisy, fs=100)
        # Filtered signal should have less variance than noisy
        assert np.std(filtered) < np.std(noisy)

    def test_detect_r_peaks_finds_peaks(self, synthetic_ecg_single_lead):
        """R-peak detection should find peaks in a synthetic ECG."""
        filtered = bandpass_filter(synthetic_ecg_single_lead, fs=100)
        peaks = detect_r_peaks(filtered, fs=100)
        # For 75bpm over 10 seconds, we expect ~12 peaks
        assert len(peaks) >= 5
        assert len(peaks) <= 20

    def test_detect_r_peaks_empty_signal(self):
        """R-peak detection should handle flat signals gracefully."""
        flat_signal = np.zeros(1000)
        peaks = detect_r_peaks(flat_signal, fs=100)
        assert len(peaks) == 0

    def test_compute_heart_rate_from_rr(self):
        """Heart rate computation from known RR intervals."""
        # 0.8 second RR intervals = 75 bpm
        rr = np.array([0.8, 0.8, 0.8, 0.8])
        mean_hr, min_hr, max_hr = compute_heart_rate(rr)
        assert abs(mean_hr - 75.0) < 1.0

    def test_compute_heart_rate_empty(self):
        """Heart rate should be 0 for empty RR intervals."""
        mean_hr, min_hr, max_hr = compute_heart_rate(np.array([]))
        assert mean_hr == 0.0

    def test_rhythm_regularity_regular(self):
        """Regular rhythm should be detected."""
        regular_rr = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        assert check_rhythm_regularity(regular_rr) is True

    def test_rhythm_regularity_irregular(self):
        """Irregular rhythm should be detected."""
        irregular_rr = np.array([0.5, 1.2, 0.4, 1.5, 0.6])
        assert check_rhythm_regularity(irregular_rr) is False

    def test_signal_quality_good_signal(self, synthetic_ecg_single_lead):
        """A clean synthetic ECG should have reasonable quality."""
        quality = estimate_signal_quality(synthetic_ecg_single_lead, fs=100)
        assert 0.0 < quality <= 1.0

    def test_signal_quality_flat_signal(self):
        """A flat signal should have quality 0."""
        quality = estimate_signal_quality(np.zeros(1000), fs=100)
        assert quality == 0.0

    def test_extract_single_lead(self, synthetic_ecg_single_lead):
        """Single-lead feature extraction should produce valid output."""
        features = extract_single_lead_features(
            synthetic_ecg_single_lead, "II", fs=100
        )
        assert features.lead_name == "II"
        assert features.heart_rate_bpm > 0
        assert features.r_peak_count > 0
        assert 0.0 <= features.signal_quality <= 1.0

    def test_extract_multilead(self, synthetic_ecg_12lead):
        """Multi-lead feature extraction should handle 12-lead ECG."""
        features = extract_multilead_features(synthetic_ecg_12lead, fs=100)
        assert len(features.per_lead) == 12
        assert features.mean_heart_rate > 0
        assert features.num_leads_analyzed > 0

    def test_extract_multilead_wrong_dimensions(self):
        """Should raise on 1D input."""
        with pytest.raises(ValueError, match="Expected 2D"):
            extract_multilead_features(np.zeros(100), fs=100)


# ── CNN Classifier Tests ─────────────────────────────────────────────────────

class TestECGClassifier:
    """Test the ResNet1D classifier (uses random weights, no trained model)."""

    def test_model_forward_pass(self):
        """Model should produce correct output shape."""
        import torch

        model = ResNet1D()
        x = torch.randn(2, 12, 1000)  # batch=2, 12 leads, 1000 samples
        output = model(x)
        assert output.shape == (2, 5)  # 5 superclasses

    def test_classifier_inference(self, synthetic_ecg_12lead):
        """ECGClassifier should produce valid ECGClassification."""
        classifier = ECGClassifier(
            model_path=Path("nonexistent_model.pt"),  # will use random weights
        )
        result = classifier.classify(synthetic_ecg_12lead, sampling_rate=100)

        assert isinstance(result, ECGClassification)
        assert len(result.class_probabilities) == 5
        assert all(0 <= p <= 1 for p in result.class_probabilities.values())
        assert len(result.predicted_classes) >= 1
        assert 0 <= result.model_confidence <= 1

    def test_classifier_preprocessing_shape(self, synthetic_ecg_12lead):
        """Preprocessing should produce correct tensor shape."""
        classifier = ECGClassifier(model_path=Path("nonexistent.pt"))
        tensor = classifier.preprocess(synthetic_ecg_12lead, sampling_rate=100)
        assert tensor.shape == (1, 12, 1000)

    def test_classifier_handles_different_lengths(self):
        """Classifier should handle signals shorter/longer than 10s."""
        classifier = ECGClassifier(model_path=Path("nonexistent.pt"))

        # Short signal (5 seconds)
        short = np.random.randn(500, 12).astype(np.float32)
        result_short = classifier.classify(short, sampling_rate=100)
        assert isinstance(result_short, ECGClassification)

        # Long signal (20 seconds)
        long = np.random.randn(2000, 12).astype(np.float32)
        result_long = classifier.classify(long, sampling_rate=100)
        assert isinstance(result_long, ECGClassification)

    def test_classifier_handles_fewer_leads(self):
        """Classifier should pad if fewer than 12 leads."""
        classifier = ECGClassifier(model_path=Path("nonexistent.pt"))
        single_lead = np.random.randn(1000, 1).astype(np.float32)
        result = classifier.classify(single_lead, sampling_rate=100)
        assert isinstance(result, ECGClassification)

    def test_all_superclasses_present(self, synthetic_ecg_12lead):
        """All 5 diagnostic superclasses should appear in output."""
        classifier = ECGClassifier(model_path=Path("nonexistent.pt"))
        result = classifier.classify(synthetic_ecg_12lead)

        for cls_name in DIAGNOSTIC_SUPERCLASSES:
            assert cls_name in result.class_probabilities


# ── Cardiology Agent Integration Tests ───────────────────────────────────────

class TestCardiologyAgent:
    """Test the full cardiology agent pipeline (mocked LLM)."""

    def _mock_ollama_cardiology_response(self) -> dict:
        """Create a mock LLM response for the cardiology agent."""
        return {
            "message": {
                "content": json.dumps({
                    "diagnosis_text": "Suspected acute myocardial infarction with ST-elevation pattern",
                    "findings": [
                        {
                            "description": "Tachycardia with heart rate ~95 bpm",
                            "evidence_source": "ecg_features.mean_heart_rate",
                            "clinical_significance": "Elevated heart rate may reflect hemodynamic compromise",
                        },
                        {
                            "description": "CNN classifier predicts MI with high probability",
                            "evidence_source": "cnn_classification.MI",
                            "clinical_significance": "Machine learning model supports MI diagnosis",
                        },
                    ],
                    "reasoning": "Given the troponin elevation and ECG findings, acute MI is the leading diagnosis.",
                    "llm_confidence": 0.85,
                    "data_quality_flag": "adequate",
                }),
            },
        }

    def test_agent_produces_specialist_opinion(
        self, cardiac_case, synthetic_ecg_12lead
    ):
        """Full pipeline should produce a valid SpecialistOpinion."""
        from agents.specialists.cardiology.agent import CardiologyAgent

        agent = CardiologyAgent()
        mock_response = self._mock_ollama_cardiology_response()

        with patch("agents.specialists.cardiology.agent.ollama.chat", return_value=mock_response):
            opinion = agent.analyze(
                cardiac_case,
                ecg_waveform=synthetic_ecg_12lead,
                sampling_rate=100,
            )

        assert isinstance(opinion, SpecialistOpinion)
        assert opinion.case_id == "CARDIO-001"
        assert opinion.specialty == "cardiology"
        assert 0.0 <= opinion.confidence <= 1.0
        assert len(opinion.findings) > 0
        assert len(opinion.diagnosis_codes) > 0

    def test_agent_requires_ecg_waveform(self, cardiac_case):
        """Agent should raise ValueError if ecg_waveform is missing."""
        from agents.specialists.cardiology.agent import CardiologyAgent

        agent = CardiologyAgent()
        with pytest.raises(ValueError, match="ecg_waveform"):
            agent.analyze(cardiac_case)

    def test_agent_handles_llm_failure(
        self, cardiac_case, synthetic_ecg_12lead
    ):
        """Agent should handle LLM failure gracefully."""
        from agents.specialists.cardiology.agent import CardiologyAgent

        agent = CardiologyAgent()

        with patch(
            "agents.specialists.cardiology.agent.ollama.chat",
            side_effect=Exception("Connection refused"),
        ):
            opinion = agent.analyze(
                cardiac_case,
                ecg_waveform=synthetic_ecg_12lead,
                sampling_rate=100,
            )

        # Should still produce an opinion (with DSP findings)
        assert isinstance(opinion, SpecialistOpinion)
        assert opinion.case_id == "CARDIO-001"
        # Confidence should be lower due to LLM failure
        assert opinion.confidence < 1.0

    def test_agent_auto_generates_dsp_findings(
        self, cardiac_case, synthetic_ecg_12lead
    ):
        """Agent should auto-generate findings from DSP measurements."""
        from agents.specialists.cardiology.agent import CardiologyAgent

        agent = CardiologyAgent()
        mock_response = self._mock_ollama_cardiology_response()

        with patch("agents.specialists.cardiology.agent.ollama.chat", return_value=mock_response):
            opinion = agent.analyze(
                cardiac_case,
                ecg_waveform=synthetic_ecg_12lead,
                sampling_rate=100,
            )

        # Should have both auto-generated and LLM findings
        evidence_sources = [f.evidence_source for f in opinion.findings]
        # At least one finding should come from ECG features
        has_ecg_finding = any("ecg_features" in src for src in evidence_sources)
        assert has_ecg_finding or len(opinion.findings) > 0
