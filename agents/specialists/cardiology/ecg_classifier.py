"""ECG 1D-CNN Classifier — ResNet1D for diagnostic superclass prediction.

A lightweight 1D ResNet trained on PTB-XL for multi-label classification
of 5 ECG diagnostic superclasses:

  NORM — Normal ECG
  MI   — Myocardial Infarction
  STTC — ST/T Changes
  CD   — Conduction Disturbance
  HYP  — Hypertrophy

Architecture: ResNet1D-18 adapted for 12-lead ECG signals.
Input:  (batch, 12, 1000) — 12 leads × 10 seconds @ 100Hz
Output: (batch, 5) — sigmoid probabilities per superclass

The classifier provides structured diagnostic predictions that are fed
to the LLM for narrative clinical reasoning (the hybrid architecture's
key value proposition).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(DIAGNOSTIC_SUPERCLASSES)
NUM_LEADS = 12
EXPECTED_SAMPLES = 1000  # 10 seconds @ 100Hz
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = Path("data/models/ecg_resnet1d.pt")


# ── Output Model ─────────────────────────────────────────────────────────────

class ECGClassification(BaseModel):
    """Structured output from the ECG classifier."""

    class_probabilities: dict[str, float] = Field(
        ...,
        description="Predicted probability for each diagnostic superclass",
    )
    predicted_classes: list[str] = Field(
        ...,
        description="Classes above the decision threshold",
    )
    model_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall model confidence (max predicted probability)",
    )


# ── ResNet1D Architecture ────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """A single residual block with 1D convolutions.

    Architecture:
        x → Conv1D → BN → ReLU → Conv1D → BN → (+skip) → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Skip connection with dimension matching
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """Lightweight 1D ResNet for 12-lead ECG classification.

    Architecture:
        Input (12, 1000)
        → Stem: Conv1D(12→64) + BN + ReLU + MaxPool
        → Block1: 2× ResBlock(64→64)
        → Block2: 2× ResBlock(64→128, stride=2)
        → Block3: 2× ResBlock(128→256, stride=2)
        → Block4: 2× ResBlock(256→512, stride=2)
        → Global Average Pool
        → FC(512→5)
        → Sigmoid (during inference)
    """

    def __init__(
        self,
        num_leads: int = NUM_LEADS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, dropout=dropout)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_classes)

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_leads, num_samples).

        Returns:
            Logits of shape (batch, num_classes). Apply sigmoid for probabilities.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


# ── Inference Interface ──────────────────────────────────────────────────────

class ECGClassifier:
    """High-level inference interface for the ECG classifier.

    Handles model loading, input preprocessing, and output formatting.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = ResNet1D()
        self.model_loaded = False

        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if model_path.exists():
            self._load_weights(model_path)
        else:
            logger.warning(
                "Model weights not found at %s — classifier will use random weights. "
                "Run train_classifier.py to train the model.",
                model_path,
            )

    def _load_weights(self, path: Path) -> None:
        """Load trained model weights."""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_loaded = True
            logger.info("Loaded ECG classifier weights from %s", path)
        except Exception as exc:
            logger.error("Failed to load model weights from %s: %s", path, exc)
            self.model_loaded = False

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, waveform: np.ndarray, sampling_rate: int = 100) -> torch.Tensor:
        """Preprocess a raw ECG waveform for model input.

        Args:
            waveform: Shape (num_samples, num_leads) or (num_leads, num_samples).
            sampling_rate: Input sampling rate (resampled to 100Hz if different).

        Returns:
            Tensor of shape (1, num_leads, expected_samples).
        """
        # Ensure shape is (num_samples, num_leads)
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        if waveform.shape[0] == NUM_LEADS and waveform.shape[1] > NUM_LEADS:
            # Likely (leads, samples) format — transpose
            waveform = waveform.T

        num_samples, num_leads = waveform.shape

        # Resample if needed
        if sampling_rate != 100:
            from scipy.signal import resample
            target_samples = int(num_samples * 100 / sampling_rate)
            waveform = resample(waveform, target_samples, axis=0)
            num_samples = target_samples

        # Pad or truncate to expected length
        if num_samples < EXPECTED_SAMPLES:
            padding = np.zeros((EXPECTED_SAMPLES - num_samples, num_leads))
            waveform = np.vstack([waveform, padding])
        elif num_samples > EXPECTED_SAMPLES:
            waveform = waveform[:EXPECTED_SAMPLES]

        # Handle lead count mismatch
        if num_leads < NUM_LEADS:
            padding = np.zeros((EXPECTED_SAMPLES, NUM_LEADS - num_leads))
            waveform = np.hstack([waveform, padding])
        elif num_leads > NUM_LEADS:
            waveform = waveform[:, :NUM_LEADS]

        # Normalize per recording: zero mean, unit variance
        mean = np.mean(waveform, axis=0, keepdims=True)
        std = np.std(waveform, axis=0, keepdims=True)
        std[std < 1e-8] = 1.0  # avoid division by zero
        waveform = (waveform - mean) / std

        # Convert to tensor: (1, num_leads, num_samples)
        tensor = torch.FloatTensor(waveform.T).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def classify(
        self, waveform: np.ndarray, sampling_rate: int = 100
    ) -> ECGClassification:
        """Run inference on a 12-lead ECG waveform.

        Args:
            waveform: Raw ECG data, shape (num_samples, num_leads).
            sampling_rate: Sampling rate in Hz.

        Returns:
            ECGClassification with per-class probabilities and predictions.
        """
        self.model.eval()
        x = self.preprocess(waveform, sampling_rate)

        logits = self.model(x)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        class_probs = {
            name: float(prob)
            for name, prob in zip(DIAGNOSTIC_SUPERCLASSES, probabilities)
        }

        predicted = [
            name for name, prob in class_probs.items()
            if prob >= self.threshold
        ]

        # If nothing above threshold, predict the most likely class
        if not predicted:
            predicted = [DIAGNOSTIC_SUPERCLASSES[int(np.argmax(probabilities))]]

        confidence = float(np.max(probabilities))

        return ECGClassification(
            class_probabilities=class_probs,
            predicted_classes=predicted,
            model_confidence=confidence,
        )
