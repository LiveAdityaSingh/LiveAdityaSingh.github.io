"""Cardiology Specialist Agent — Hybrid DSP + CNN + LLM Pipeline.

The cardiology agent is the most technically differentiated specialist in
the MLTVA pipeline. It combines three analysis modalities:

  1. DSP Pipeline → structured ECG features (QRS, ST, QT, heart rate)
  2. 1D-CNN → diagnostic class probabilities (NORM, MI, STTC, CD, HYP)
  3. LLM → narrative clinical reasoning from numerical evidence

The LLM receives numerical findings (not raw waveform data), which is
the hybrid architecture's key value proposition: the CNN handles pattern
recognition, the DSP provides interpretable measurements, and the LLM
synthesises both into clinical reasoning.

This agent implements the BaseSpecialist interface and produces a
SpecialistOpinion that feeds into the Consolidator (M5) and Auditor (M6).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import ollama
from pydantic import ValidationError

from schemas.fhir_input import PatientCase
from schemas.specialist_opinion import Finding, SpecialistOpinion
from agents.specialists.base_specialist import BaseSpecialist
from agents.specialists.cardiology.ecg_features import (
    MultiLeadECGFeatures,
    extract_multilead_features,
)
from agents.specialists.cardiology.ecg_classifier import (
    ECGClassification,
    ECGClassifier,
    DIAGNOSTIC_SUPERCLASSES,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openbiollm"
MAX_SCHEMA_RETRIES = 2

# ICD-10 codes for PTB-XL diagnostic superclasses
SUPERCLASS_ICD10_MAP: dict[str, list[str]] = {
    "NORM": ["Z01.810"],              # Encounter for ECG: normal finding
    "MI":   ["I21.9", "I25.2"],       # Acute MI (unspecified), old MI
    "STTC": ["I25.6", "R94.31"],      # Silent myocardial ischemia, abnormal ECG
    "CD":   ["I44.0", "I45.9"],       # AV block, conduction disorder unspecified
    "HYP":  ["I51.7"],                # Cardiomegaly / hypertrophy
}

# Confidence weighting: CNN is more calibratable than LLM self-assessment
CNN_CONFIDENCE_WEIGHT = 0.7
LLM_CONFIDENCE_WEIGHT = 0.3


SYSTEM_PROMPT = """\
You are a cardiology specialist in a clinical decision-support system.
You are analysing ECG findings for a patient case.

CRITICAL CONSTRAINTS:
- You are a DECISION-SUPPORT tool, NOT an autonomous diagnostic system.
  A human clinician will always review your output.
- Base your reasoning ONLY on the numerical ECG findings, CNN predictions,
  and patient data provided. Do NOT fabricate or assume additional data.
- Be specific about which ECG leads and measurements support each finding.

INPUT DATA:
You will receive:
1. Patient demographics and symptoms (from the EMR)
2. DSP-extracted ECG features (heart rate, QRS duration, ST deviation, etc.)
3. CNN classifier predictions (diagnostic superclass probabilities)

OUTPUT FORMAT:
Respond with a JSON object matching this exact structure:
{
  "diagnosis_text": "<your clinical diagnosis/assessment>",
  "findings": [
    {
      "description": "<what you found>",
      "evidence_source": "<which measurement or prediction supports this>",
      "clinical_significance": "<why it matters clinically>"
    }
  ],
  "reasoning": "<your step-by-step clinical reasoning>",
  "llm_confidence": <0.0 to 1.0>,
  "data_quality_flag": "adequate|limited|insufficient"
}

RULES:
1. Include 2–6 findings, each tied to a specific piece of evidence.
2. Your reasoning should follow standard cardiology logic: correlate ST
   changes with coronary territories, assess rhythm, consider clinical
   context (symptoms, risk factors).
3. Assess data quality honestly — if the ECG signal quality is poor or
   key leads are missing, flag it.
"""


# ── Helper Functions ─────────────────────────────────────────────────────────

def _build_ecg_context(
    case: PatientCase,
    ecg_features: MultiLeadECGFeatures,
    classification: ECGClassification,
) -> str:
    """Build the user prompt with patient data + ECG analysis results."""

    # Patient context
    obs_text = "\n".join(
        f"  - [{o.code}] {o.display}: {o.value}"
        + (f" {o.unit}" if o.unit else "")
        for o in case.observations
    )
    symptoms_text = "\n".join(f"  - {s}" for s in case.symptoms)
    notes_text = "\n".join(f"  - {n}" for n in case.notes) if case.notes else "  (none)"

    # ECG features summary
    feature_lines = []
    for lead in ecg_features.per_lead:
        parts = [f"Lead {lead.lead_name}:"]
        if lead.heart_rate_bpm > 0:
            parts.append(f"HR={lead.heart_rate_bpm:.0f}bpm")
        if lead.qrs_duration_ms is not None:
            parts.append(f"QRS={lead.qrs_duration_ms:.0f}ms")
        if lead.qt_interval_ms is not None:
            parts.append(f"QT={lead.qt_interval_ms:.0f}ms")
        if lead.qtc_bazett_ms is not None:
            parts.append(f"QTc={lead.qtc_bazett_ms:.0f}ms")
        if lead.st_deviation_mv is not None:
            parts.append(f"ST={lead.st_deviation_mv:+.3f}mV")
        parts.append(f"quality={lead.signal_quality:.2f}")
        feature_lines.append("  " + " | ".join(parts))

    features_text = "\n".join(feature_lines)

    # CNN predictions
    sorted_probs = sorted(
        classification.class_probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    cnn_text = "\n".join(
        f"  - {name}: {prob:.1%}" for name, prob in sorted_probs
    )

    return f"""\
PATIENT CASE (ID: {case.case_id})
- Age: {case.age}, Sex: {case.sex}
- Chief complaint: {case.chief_complaint}

Symptoms:
{symptoms_text}

Observations (labs/vitals):
{obs_text}

Clinical notes:
{notes_text}

──── ECG ANALYSIS RESULTS ────

Overall: HR={ecg_features.mean_heart_rate:.0f}bpm | \
Rhythm={'regular' if ecg_features.rhythm_regular else 'IRREGULAR'} | \
Signal quality={ecg_features.overall_signal_quality:.2f} | \
Leads analyzed={ecg_features.num_leads_analyzed}

Per-lead features:
{features_text}

CNN Diagnostic Predictions:
{cnn_text}
  → Predicted classes: {', '.join(classification.predicted_classes)}
  → Model confidence: {classification.model_confidence:.1%}

Provide your cardiology specialist assessment based on these findings.\
"""


def _get_cardiology_json_schema() -> dict[str, Any]:
    """JSON schema for Ollama constrained decoding of the cardiology response."""
    return {
        "type": "object",
        "properties": {
            "diagnosis_text": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "evidence_source": {"type": "string"},
                        "clinical_significance": {"type": "string"},
                    },
                    "required": ["description", "evidence_source", "clinical_significance"],
                },
            },
            "reasoning": {"type": "string"},
            "llm_confidence": {"type": "number"},
            "data_quality_flag": {"type": "string"},
        },
        "required": ["diagnosis_text", "findings", "reasoning", "llm_confidence", "data_quality_flag"],
    }


# ── Cardiology Agent ─────────────────────────────────────────────────────────

class CardiologyAgent(BaseSpecialist):
    """Cardiology specialist using hybrid DSP + CNN + LLM pipeline.

    Usage:
        agent = CardiologyAgent()
        opinion = agent.analyze(
            case=patient_case,
            ecg_waveform=ecg_array,      # shape (samples, 12)
            sampling_rate=100,
        )
    """

    specialty = "cardiology"

    def __init__(
        self,
        model: str | None = None,
        classifier_path: Path | None = None,
    ):
        self.model_name = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.classifier = ECGClassifier(model_path=classifier_path)

    def analyze(
        self, case: PatientCase, **domain_data
    ) -> SpecialistOpinion:
        """Analyse a patient case with ECG data.

        Required domain_data:
            ecg_waveform (np.ndarray): Raw ECG, shape (samples, num_leads).
            sampling_rate (int, optional): Defaults to 100 Hz.
        """
        self._validate_case(case)

        # Extract domain-specific inputs
        ecg_waveform = domain_data.get("ecg_waveform")
        if ecg_waveform is None:
            raise ValueError(
                "CardiologyAgent requires 'ecg_waveform' in domain_data. "
                "Provide a numpy array of shape (num_samples, num_leads)."
            )
        sampling_rate = domain_data.get("sampling_rate", 100)

        logger.info("Cardiology agent analyzing case %s", case.case_id)

        # Step 1: DSP feature extraction
        logger.info("Step 1/3: Extracting ECG features (DSP)")
        ecg_features = extract_multilead_features(
            ecg_waveform, fs=sampling_rate
        )

        # Step 2: CNN classification
        logger.info("Step 2/3: Running CNN classifier")
        classification = self.classifier.classify(ecg_waveform, sampling_rate)

        # Step 3: LLM reasoning
        logger.info("Step 3/3: LLM clinical reasoning")
        llm_result = self._llm_reasoning(case, ecg_features, classification)

        # Combine into SpecialistOpinion
        return self._build_opinion(
            case, ecg_features, classification, llm_result
        )

    def _llm_reasoning(
        self,
        case: PatientCase,
        ecg_features: MultiLeadECGFeatures,
        classification: ECGClassification,
    ) -> dict[str, Any]:
        """Call the LLM for narrative clinical reasoning."""

        user_prompt = _build_ecg_context(case, ecg_features, classification)
        last_error: Exception | None = None

        for attempt in range(1, MAX_SCHEMA_RETRIES + 2):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    format=_get_cardiology_json_schema(),
                )

                raw = response["message"]["content"]
                parsed = json.loads(raw)

                # Validate required fields
                assert "diagnosis_text" in parsed
                assert "findings" in parsed
                assert "reasoning" in parsed

                return parsed

            except (json.JSONDecodeError, KeyError, AssertionError) as exc:
                last_error = exc
                logger.warning(
                    "LLM output parsing failed (attempt %d) for case %s: %s",
                    attempt, case.case_id, exc,
                )
                continue

            except Exception as exc:
                last_error = exc
                logger.error(
                    "LLM call failed (attempt %d) for case %s: %s",
                    attempt, case.case_id, exc,
                )
                continue

        # All retries exhausted — return a minimal fallback
        logger.error(
            "LLM reasoning failed after %d attempts for case %s: %s",
            MAX_SCHEMA_RETRIES + 1, case.case_id, last_error,
        )
        return {
            "diagnosis_text": f"[LLM REASONING FAILED: {last_error}]",
            "findings": [],
            "reasoning": "Unable to generate clinical reasoning due to LLM failure.",
            "llm_confidence": 0.0,
            "data_quality_flag": "insufficient",
        }

    def _build_opinion(
        self,
        case: PatientCase,
        ecg_features: MultiLeadECGFeatures,
        classification: ECGClassification,
        llm_result: dict[str, Any],
    ) -> SpecialistOpinion:
        """Combine all analysis components into a SpecialistOpinion."""

        # Map predicted classes to ICD-10 codes
        icd10_codes: list[str] = []
        for cls_name in classification.predicted_classes:
            icd10_codes.extend(SUPERCLASS_ICD10_MAP.get(cls_name, []))

        # Combine findings from LLM + auto-generated from ECG analysis
        findings: list[Finding] = []

        # Auto-generated findings from DSP
        if ecg_features.mean_heart_rate > 0:
            hr = ecg_features.mean_heart_rate
            if hr > 100:
                findings.append(Finding(
                    description=f"Tachycardia: heart rate {hr:.0f} bpm",
                    evidence_source="ecg_features.mean_heart_rate",
                    clinical_significance="Heart rate above 100 bpm suggests tachycardia",
                ))
            elif hr < 60:
                findings.append(Finding(
                    description=f"Bradycardia: heart rate {hr:.0f} bpm",
                    evidence_source="ecg_features.mean_heart_rate",
                    clinical_significance="Heart rate below 60 bpm suggests bradycardia",
                ))

        if not ecg_features.rhythm_regular:
            findings.append(Finding(
                description="Irregular rhythm detected",
                evidence_source="ecg_features.rhythm_regular",
                clinical_significance="May indicate atrial fibrillation or other arrhythmia",
            ))

        # ST deviation findings per lead
        for lead_feat in ecg_features.per_lead:
            if lead_feat.st_deviation_mv is not None:
                st = lead_feat.st_deviation_mv
                if abs(st) > 0.1:  # >0.1mV is clinically significant
                    direction = "elevation" if st > 0 else "depression"
                    findings.append(Finding(
                        description=f"ST {direction} of {abs(st):.3f} mV in lead {lead_feat.lead_name}",
                        evidence_source=f"ecg_features.{lead_feat.lead_name}.st_deviation_mv",
                        clinical_significance=(
                            f"ST {direction} in {lead_feat.lead_name} "
                            f"{'may indicate acute ischaemia/infarction' if st > 0 else 'may indicate ischaemia or strain'}"
                        ),
                    ))

        # LLM-generated findings
        for f in llm_result.get("findings", []):
            try:
                findings.append(Finding(
                    description=f["description"],
                    evidence_source=f["evidence_source"],
                    clinical_significance=f["clinical_significance"],
                ))
            except (KeyError, ValidationError) as exc:
                logger.warning("Skipping invalid LLM finding: %s", exc)

        # Compute combined confidence
        llm_confidence = float(llm_result.get("llm_confidence", 0.5))
        llm_confidence = max(0.0, min(1.0, llm_confidence))

        combined_confidence = (
            CNN_CONFIDENCE_WEIGHT * classification.model_confidence
            + LLM_CONFIDENCE_WEIGHT * llm_confidence
        )

        return SpecialistOpinion(
            case_id=case.case_id,
            specialty=self.specialty,
            diagnosis_codes=icd10_codes if icd10_codes else ["R94.31"],  # abnormal ECG
            diagnosis_text=llm_result.get("diagnosis_text", "Unable to determine diagnosis"),
            findings=findings,
            confidence=combined_confidence,
            reasoning=llm_result.get("reasoning", "No reasoning available"),
            data_quality_flag=llm_result.get("data_quality_flag", "adequate"),
        )
