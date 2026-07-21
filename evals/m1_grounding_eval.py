"""M1 Grounding Evaluation — Standalone precision/recall report.

Runs the grounding checker against a labeled test set and reports:
  - Per-claim accuracy
  - Precision, recall, F1 for grounded vs. ungrounded classification
  - Confusion matrix
  - Per-case breakdown

Usage:
    python -m evals.m1_grounding_eval

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
CASES_FILE = DATA_DIR / "sample_cases.json"
LABELS_FILE = DATA_DIR / "m1_eval_labels.json"


def load_cases() -> dict[str, PatientCase]:
    """Load patient cases keyed by case_id."""
    with open(CASES_FILE) as f:
        raw = json.load(f)
    return {c["case_id"]: PatientCase(**c) for c in raw}


def load_labels() -> list[dict]:
    """Load labeled evaluation data."""
    with open(LABELS_FILE) as f:
        return json.load(f)


def run_eval() -> dict:
    """Run the grounding evaluation and return metrics."""
    cases = load_cases()
    label_groups = load_labels()

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


def print_report(results: dict) -> None:
    """Print a formatted evaluation report to stdout."""
    m = results["metrics"]

    print("\n" + "=" * 70)
    print("  MLTVA — M1 GROUNDING CHECKER EVALUATION REPORT")
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
    results = run_eval()
    print_report(results)

    # Exit with non-zero code if acceptance criterion not met
    if results["metrics"]["accuracy"] < 0.90:
        sys.exit(1)


if __name__ == "__main__":
    main()
