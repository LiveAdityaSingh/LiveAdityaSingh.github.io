"""M1 Grounding Evaluation — Standalone precision/recall report.

Runs the grounding checker against a labeled test set and reports:
  - Per-claim accuracy
  - Precision, recall, F1 for grounded vs. ungrounded classification
  - Confusion matrix
  - Per-case breakdown

Usage:
    python -m evals.m1_grounding_eval                  # default: MIMIC data
    python -m evals.m1_grounding_eval --source mimic    # MIMIC-IV sourced cases
    python -m evals.m1_grounding_eval --source synthetic # HuggingFace synthetic cases

This script has NO dependency on M2+ and tests the grounding checker
in isolation using pre-labeled claims (not LLM-generated ones).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from schemas.fhir_input import PatientCase
from schemas.hypothesis import Claim
from agents.primary_llm.grounding_checker import ground_single_claim

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Data source file mappings
DATA_SOURCES = {
    "synthetic": {
        "cases": DATA_DIR / "sample_cases.json",
        "labels": DATA_DIR / "m1_eval_labels.json",
    },
    "mimic": {
        "cases": DATA_DIR / "mimic_sample_cases.json",
        "labels": DATA_DIR / "m1_eval_labels_mimic.json",
    },
}


def load_cases(source: str = "mimic") -> dict[str, PatientCase]:
    """Load patient cases keyed by case_id."""
    cases_file = DATA_SOURCES[source]["cases"]
    if not cases_file.exists():
        # Fall back to synthetic if MIMIC cases not yet generated
        if source == "mimic":
            logger.warning(
                "MIMIC cases not found at %s, falling back to synthetic",
                cases_file,
            )
            cases_file = DATA_SOURCES["synthetic"]["cases"]
        else:
            raise FileNotFoundError(f"Cases file not found: {cases_file}")

    with open(cases_file) as f:
        raw = json.load(f)
    return {c["case_id"]: PatientCase(**c) for c in raw}


def load_labels(source: str = "mimic") -> list[dict]:
    """Load labeled evaluation data."""
    labels_file = DATA_SOURCES[source]["labels"]
    if not labels_file.exists():
        # Fall back to synthetic if MIMIC labels not yet annotated
        if source == "mimic":
            logger.warning(
                "MIMIC labels not found at %s, falling back to synthetic",
                labels_file,
            )
            labels_file = DATA_SOURCES["synthetic"]["labels"]
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

    with open(labels_file) as f:
        return json.load(f)


def run_eval(source: str = "mimic") -> dict:
    """Run the grounding evaluation and return metrics."""
    cases = load_cases(source)
    label_groups = load_labels(source)

    tp = fp = tn = fn = 0
    results_detail: list[dict] = []

    for group in label_groups:
        case_id = group["case_id"]
        case = cases.get(case_id)
        if case is None:
            logger.warning("Case %s not found in sample_cases.json, skipping", case_id)
            continue

        for claim_data in group["claims"]:
            claim = Claim(
                text=claim_data["text"],
                supporting_evidence_ref=claim_data.get("supporting_evidence_ref"),
            )
            expected = claim_data["expected_grounded"]

            result = ground_single_claim(claim, case)
            predicted = result.grounded

            # Confusion matrix update
            if expected and predicted:
                tp += 1
            elif expected and not predicted:
                fn += 1
            elif not expected and predicted:
                fp += 1
            else:
                tn += 1

            match = predicted == expected
            results_detail.append({
                "case_id": case_id,
                "claim_text": claim.text[:80],
                "expected": expected,
                "predicted": predicted,
                "match": match,
                "reason": result.reason,
            })

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "total_claims": total,
        "correct": tp + tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }

    return {"metrics": metrics, "details": results_detail}


def print_report(results: dict, source: str = "mimic") -> None:
    """Print a formatted evaluation report to stdout."""
    m = results["metrics"]

    print("\n" + "=" * 70)
    print("  MLTVA — M1 GROUNDING CHECKER EVALUATION REPORT")
    print(f"  Data source: {source.upper()}")
    print("=" * 70)

    print(f"\n  Total claims evaluated:  {m['total_claims']}")
    print(f"  Correctly classified:    {m['correct']}/{m['total_claims']}")
    print(f"  Accuracy:                {m['accuracy']:.1%}")
    print(f"\n  Precision (grounded):    {m['precision']:.1%}")
    print(f"  Recall (grounded):       {m['recall']:.1%}")
    print(f"  F1 Score:                {m['f1']:.1%}")

    cm = m["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Grounded    Ungrounded")
    print(f"  Actual Grounded    {cm['TP']:>3}          {cm['FN']:>3}")
    print(f"  Actual Ungrounded  {cm['FP']:>3}          {cm['TN']:>3}")

    # Detailed results
    print(f"\n{'-' * 70}")
    print("  DETAILED RESULTS")
    print(f"{'-' * 70}")

    mismatches = [d for d in results["details"] if not d["match"]]
    if mismatches:
        print(f"\n  ! {len(mismatches)} MISMATCHES:\n")
        for d in mismatches:
            print(f"  [{d['case_id']}] \"{d['claim_text']}\"")
            print(f"    Expected: {'grounded' if d['expected'] else 'ungrounded'}")
            print(f"    Got:      {'grounded' if d['predicted'] else 'ungrounded'}")
            print(f"    Reason:   {d['reason']}")
            print()
    else:
        print("\n  [OK] All claims classified correctly!")

    print("=" * 70)

    # Check acceptance criterion: >=90% accuracy
    if m["accuracy"] >= 0.90:
        print("  [PASS] Accuracy >= 90% acceptance criterion met")
    else:
        print("  [FAIL] Accuracy < 90% -- grounding checker needs improvement")

    print("=" * 70 + "\n")


def main():
    """Entry point for the evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="M1 Grounding Evaluation")
    parser.add_argument(
        "--source",
        choices=["mimic", "synthetic"],
        default="mimic",
        help="Data source: mimic (default) or synthetic",
    )
    args = parser.parse_args()

    results = run_eval(source=args.source)
    print_report(results, source=args.source)

    # Exit with non-zero code if acceptance criterion not met
    if results["metrics"]["accuracy"] < 0.90:
        sys.exit(1)


if __name__ == "__main__":
    main()
