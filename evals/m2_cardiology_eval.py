"""M2 Cardiology Evaluation — Standalone accuracy report.

Evaluates the cardiology specialist agent:
  - CNN classifier metrics (AUROC per class) on held-out test data
  - DSP feature extraction on synthetic ECGs with known parameters
  - Full pipeline on sample cardiac cases (requires Ollama)

Usage:
    python -m evals.m2_cardiology_eval              # CNN + DSP only
    python -m evals.m2_cardiology_eval --full        # includes LLM pipeline

This script has NO dependency on M3+ and tests M2 in isolation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def eval_dsp_pipeline() -> dict:
    """Evaluate the DSP pipeline on synthetic ECGs with known parameters."""
    from agents.specialists.cardiology.ecg_features import (
        extract_single_lead_features,
        extract_multilead_features,
    )

    print("\n── DSP Feature Extraction Evaluation ──")

    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Known heart rate recovery
    fs = 100
    t = np.arange(1000) / fs
    hr_targets = [60, 75, 100, 120]

    for target_hr in hr_targets:
        rr = 60.0 / target_hr
        signal = np.zeros(1000, dtype=np.float32)
        for beat in np.arange(0, 10, rr):
            signal += 1.0 * np.exp(-((t - beat) ** 2) / (2 * 0.01 ** 2))
        signal += np.random.normal(0, 0.02, 1000).astype(np.float32)

        features = extract_single_lead_features(signal, "II", fs=fs)
        hr_error = abs(features.heart_rate_bpm - target_hr)
        passed = hr_error < 10  # within 10 bpm

        results["tests"].append({
            "test": f"HR recovery (target={target_hr}bpm)",
            "expected": target_hr,
            "got": round(features.heart_rate_bpm, 1),
            "error": round(hr_error, 1),
            "passed": passed,
        })

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Test 2: 12-lead extraction doesn't crash
    ecg_12lead = np.random.randn(1000, 12).astype(np.float32)
    try:
        multilead = extract_multilead_features(ecg_12lead, fs=100)
        results["tests"].append({
            "test": "12-lead extraction",
            "expected": "no crash, 12 leads analyzed",
            "got": f"{multilead.num_leads_analyzed} leads",
            "passed": True,
        })
        results["passed"] += 1
    except Exception as exc:
        results["tests"].append({
            "test": "12-lead extraction",
            "expected": "no crash",
            "got": str(exc),
            "passed": False,
        })
        results["failed"] += 1

    return results


def eval_cnn_classifier(use_synthetic: bool = True) -> dict:
    """Evaluate the CNN classifier on test data."""
    from agents.specialists.cardiology.ecg_classifier import ECGClassifier
    from data.ptbxl_loader import (
        generate_synthetic_ecg_data,
        get_train_val_test_split,
        SUPERCLASSES,
    )

    print("\n── CNN Classifier Evaluation ──")

    # Load data
    if use_synthetic:
        X, y, folds = generate_synthetic_ecg_data(num_samples=100)
    else:
        from data.ptbxl_loader import load_ptbxl_from_huggingface
        X, y, folds = load_ptbxl_from_huggingface()

    splits = get_train_val_test_split(X, y, folds)
    X_test, y_test = splits["test"]

    if len(X_test) == 0:
        return {"error": "No test data available", "tests": []}

    # Load classifier
    model_path = Path("data/models/ecg_resnet1d.pt")
    classifier = ECGClassifier(model_path=model_path)

    all_probs = []
    all_labels = []

    for i in range(len(X_test)):
        result = classifier.classify(X_test[i], sampling_rate=100)
        probs = [result.class_probabilities[c] for c in SUPERCLASSES]
        all_probs.append(probs)
        all_labels.append(y_test[i])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute metrics
    results = {"per_class": {}, "tests": []}

    try:
        from sklearn.metrics import roc_auc_score

        for i, cls in enumerate(SUPERCLASSES):
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                results["per_class"][cls] = round(auroc, 4)
            else:
                results["per_class"][cls] = "N/A (single class)"

        valid_aurocs = [v for v in results["per_class"].values() if isinstance(v, float)]
        results["macro_auroc"] = round(np.mean(valid_aurocs), 4) if valid_aurocs else 0.0
    except ImportError:
        results["macro_auroc"] = "sklearn not available"

    results["model_loaded"] = classifier.model_loaded
    results["num_test_samples"] = len(X_test)

    return results


def eval_full_pipeline(case_data: list[dict] | None = None) -> dict:
    """Evaluate the full cardiology agent pipeline (requires Ollama)."""
    from schemas.fhir_input import PatientCase
    from agents.specialists.cardiology.agent import CardiologyAgent

    print("\n── Full Pipeline Evaluation ──")

    if case_data is None:
        # Use a default cardiac case
        case_data = [{
            "case_id": "EVAL-CARDIO-001",
            "age": 58,
            "sex": "M",
            "chief_complaint": "Chest pain radiating to left arm",
            "symptoms": ["chest pain", "diaphoresis", "shortness of breath"],
            "observations": [
                {"code": "6598-7", "display": "Troponin T", "value": "0.85", "unit": "ng/mL"},
                {"code": "8867-4", "display": "Heart rate", "value": "110", "unit": "bpm"},
            ],
            "notes": ["30-year smoking history. Previous angioplasty in 2019."],
        }]

    agent = CardiologyAgent()
    results = {"cases": []}

    for cd in case_data:
        case = PatientCase(**cd)

        # Generate synthetic ECG for this case
        rng = np.random.RandomState(hash(case.case_id) % (2**31))
        ecg = rng.randn(1000, 12).astype(np.float32) * 0.5

        try:
            opinion = agent.analyze(case, ecg_waveform=ecg, sampling_rate=100)
            results["cases"].append({
                "case_id": case.case_id,
                "diagnosis": opinion.diagnosis_text[:100],
                "confidence": round(opinion.confidence, 3),
                "num_findings": len(opinion.findings),
                "icd10_codes": opinion.diagnosis_codes,
                "data_quality": opinion.data_quality_flag,
                "status": "OK",
            })
        except Exception as exc:
            results["cases"].append({
                "case_id": case.case_id,
                "status": f"FAILED: {exc}",
            })

    return results


def print_report(dsp_results: dict, cnn_results: dict, pipeline_results: dict | None) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  MLTVA — M2 CARDIOLOGY SPECIALIST EVALUATION")
    print("=" * 60)

    # DSP Results
    print("\n  DSP FEATURE EXTRACTION")
    print("  " + "-" * 40)
    for test in dsp_results["tests"]:
        status = "✓" if test["passed"] else "✗"
        print(f"  [{status}] {test['test']}")
        if not test["passed"]:
            print(f"      Expected: {test['expected']}, Got: {test['got']}")

    print(f"\n  DSP: {dsp_results['passed']}/{dsp_results['passed'] + dsp_results['failed']} tests passed")

    # CNN Results
    print("\n  CNN CLASSIFIER")
    print("  " + "-" * 40)
    print(f"  Model loaded: {cnn_results.get('model_loaded', 'N/A')}")
    print(f"  Test samples: {cnn_results.get('num_test_samples', 'N/A')}")

    if "per_class" in cnn_results:
        print(f"  Macro AUROC: {cnn_results.get('macro_auroc', 'N/A')}")
        for cls, auroc in cnn_results.get("per_class", {}).items():
            print(f"    {cls:5s}: {auroc}")

    # Pipeline Results
    if pipeline_results and "cases" in pipeline_results:
        print("\n  FULL PIPELINE")
        print("  " + "-" * 40)
        for case in pipeline_results["cases"]:
            print(f"  Case {case['case_id']}: {case.get('status', 'UNKNOWN')}")
            if case.get("status") == "OK":
                print(f"    Diagnosis: {case.get('diagnosis', 'N/A')}")
                print(f"    Confidence: {case.get('confidence', 'N/A')}")
                print(f"    Findings: {case.get('num_findings', 0)}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="M2 Cardiology evaluation")
    parser.add_argument("--full", action="store_true", help="Include full pipeline eval (requires Ollama)")
    args = parser.parse_args()

    dsp_results = eval_dsp_pipeline()
    cnn_results = eval_cnn_classifier()

    pipeline_results = None
    if args.full:
        try:
            pipeline_results = eval_full_pipeline()
        except Exception as exc:
            print(f"\n  [WARN] Full pipeline eval failed: {exc}")

    print_report(dsp_results, cnn_results, pipeline_results)


if __name__ == "__main__":
    main()
