"""Train the ResNet1D ECG classifier on PTB-XL.

Standalone training script that:
  1. Loads PTB-XL data (HuggingFace or synthetic fallback)
  2. Trains a ResNet1D on 5 diagnostic superclasses
  3. Evaluates on test fold (AUROC, per-class accuracy)
  4. Saves the best model checkpoint

Usage:
    python -m agents.specialists.cardiology.train_classifier
    python -m agents.specialists.cardiology.train_classifier --synthetic
    python -m agents.specialists.cardiology.train_classifier --epochs 50

This script has NO dependency on M3+ and trains the classifier in isolation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from data.ptbxl_loader import (
    load_ptbxl_from_huggingface,
    generate_synthetic_ecg_data,
    get_train_val_test_split,
    SUPERCLASSES,
)
from agents.specialists.cardiology.ecg_classifier import ResNet1D, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_SAVE_DIR = Path("data/models")
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "ecg_resnet1d.pt"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())

        total_loss += loss.item()
        num_batches += 1

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Per-class AUROC
    from sklearn.metrics import roc_auc_score

    per_class_auroc = {}
    macro_auroc = 0.0
    valid_classes = 0

    for i, class_name in enumerate(SUPERCLASSES):
        if len(np.unique(all_labels[:, i])) > 1:
            auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_class_auroc[class_name] = auroc
            macro_auroc += auroc
            valid_classes += 1
        else:
            per_class_auroc[class_name] = float("nan")

    macro_auroc = macro_auroc / max(valid_classes, 1)

    # Per-class accuracy (threshold=0.5)
    predictions = (all_probs >= 0.5).astype(float)
    per_class_accuracy = {}
    for i, class_name in enumerate(SUPERCLASSES):
        acc = np.mean(predictions[:, i] == all_labels[:, i])
        per_class_accuracy[class_name] = acc

    return {
        "loss": total_loss / max(num_batches, 1),
        "macro_auroc": macro_auroc,
        "per_class_auroc": per_class_auroc,
        "per_class_accuracy": per_class_accuracy,
        "num_samples": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ECG ResNet1D classifier")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data only")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Training on device: %s", device)

    # ── Load Data ────────────────────────────────────────────────────────
    if args.synthetic:
        logger.info("Using synthetic ECG data (--synthetic flag)")
        X, y, folds = generate_synthetic_ecg_data(num_samples=500)
    else:
        X, y, folds = load_ptbxl_from_huggingface()

    splits = get_train_val_test_split(X, y, folds)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    logger.info(
        "Data splits: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Create DataLoaders ───────────────────────────────────────────────
    # Transpose from (N, samples, leads) to (N, leads, samples) for Conv1D
    train_ds = TensorDataset(
        torch.FloatTensor(X_train.transpose(0, 2, 1)),
        torch.FloatTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val.transpose(0, 2, 1)),
        torch.FloatTensor(y_val),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test.transpose(0, 2, 1)),
        torch.FloatTensor(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, Loss, Optimizer ───────────────────────────────────────────
    model = ResNet1D().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model has %d parameters (%.2f MB)", num_params, num_params * 4 / 1e6)

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_auroc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_auroc = val_metrics["macro_auroc"]

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_AUROC=%.4f%s",
            epoch, args.epochs, train_loss, val_metrics["loss"], val_auroc,
            " ★" if val_auroc > best_val_auroc else "",
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    logger.info(
        "Best validation AUROC: %.4f at epoch %d (saved to %s)",
        best_val_auroc, best_epoch, MODEL_SAVE_PATH,
    )

    # ── Test Evaluation ──────────────────────────────────────────────────
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("  ECG CLASSIFIER — TEST SET EVALUATION")
    print("=" * 60)
    print(f"\n  Macro-averaged AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Test loss:            {test_metrics['loss']:.4f}")
    print(f"  Test samples:         {test_metrics['num_samples']}")

    print(f"\n  Per-class AUROC:")
    for cls, auroc in test_metrics["per_class_auroc"].items():
        status = "✓" if auroc >= 0.85 else "✗" if not np.isnan(auroc) else "?"
        print(f"    [{status}] {cls:5s}: {auroc:.4f}")

    print(f"\n  Per-class Accuracy:")
    for cls, acc in test_metrics["per_class_accuracy"].items():
        print(f"    {cls:5s}: {acc:.1%}")

    target = 0.85
    if test_metrics["macro_auroc"] >= target:
        print(f"\n  [PASS] Macro AUROC >= {target:.0%} acceptance criterion met")
    else:
        print(f"\n  [WARN] Macro AUROC < {target:.0%} — may need more training or data")

    print("=" * 60)


if __name__ == "__main__":
    main()
