"""Shared utilities for reproducibility, metrics, and checkpoint handling."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch_runtime(allow_tf32: bool = False, deterministic: bool = False) -> None:
    """Configure backend knobs that affect speed and determinism."""

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def ensure_directory(path: Path) -> None:
    """Create a directory and all missing parents if necessary."""

    path.mkdir(parents=True, exist_ok=True)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the compute device from a CLI or configuration string."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def empty_confusion_matrix() -> np.ndarray:
    """Return an empty 2x2 confusion matrix for binary segmentation."""

    return np.zeros((2, 2), dtype=np.int64)


def _ensure_class_shape(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze a single-channel tensor to class indices when needed."""

    if tensor.ndim >= 4 and tensor.size(1) == 1:
        return tensor.squeeze(1)
    return tensor


def update_confusion_matrix(
    confusion_matrix: np.ndarray,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> None:
    """Accumulate binary classification counts into a 2x2 matrix."""

    predictions = _ensure_class_shape(predictions).to(torch.int64)
    targets = _ensure_class_shape(targets).to(torch.int64)
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return

    flat_predictions = predictions[valid_mask].reshape(-1).cpu().numpy()
    flat_targets = targets[valid_mask].reshape(-1).cpu().numpy()
    encoded = flat_targets * 2 + flat_predictions
    counts = np.bincount(encoded, minlength=4)
    confusion_matrix[0, 0] += int(counts[0])
    confusion_matrix[0, 1] += int(counts[1])
    confusion_matrix[1, 0] += int(counts[2])
    confusion_matrix[1, 1] += int(counts[3])


def predict_binary_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert segmentation logits into binary class predictions."""

    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).to(torch.int64)


def compute_binary_metrics(confusion_matrix: np.ndarray) -> dict[str, float]:
    """Compute IoU and accuracy metrics from a binary confusion matrix."""

    tn = float(confusion_matrix[0, 0])
    fp = float(confusion_matrix[0, 1])
    fn = float(confusion_matrix[1, 0])
    tp = float(confusion_matrix[1, 1])

    background_denom = tn + fp + fn
    foreground_denom = tp + fp + fn
    total = tn + fp + fn + tp

    background_iou = tn / background_denom if background_denom > 0 else 0.0
    foreground_iou = tp / foreground_denom if foreground_denom > 0 else 0.0
    mean_iou = (background_iou + foreground_iou) / 2.0
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "background_iou": background_iou,
        "foreground_iou": foreground_iou,
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def format_confusion_matrix(confusion_matrix: np.ndarray) -> str:
    """Format the confusion matrix for logging output."""

    return (
        "Confusion matrix (rows=target, cols=prediction)\n"
        f"[[{confusion_matrix[0, 0]:8d} {confusion_matrix[0, 1]:8d}]\n"
        f" [{confusion_matrix[1, 0]:8d} {confusion_matrix[1, 1]:8d}]]"
    )


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Persist a checkpoint dictionary to disk."""

    ensure_directory(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    return torch.load(path, map_location=map_location)
