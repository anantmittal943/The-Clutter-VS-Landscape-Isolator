"""Centralized configuration for the Clutter vs. Landscape isolator project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by training and inference."""

    train_image_dir: Path = Path("data/train/images")
    train_mask_dir: Path = Path("data/train/masks")
    val_image_dir: Path = Path("data/val/images")
    val_mask_dir: Path = Path("data/val/masks")
    test_image_dir: Path = Path("data/test/images")
    test_mask_dir: Path = Path("data/test/masks")
    output_dir: Path = Path("artifacts")
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    log_dir: Path = Path("artifacts/logs")
    predictions_dir: Path = Path("artifacts/predictions")


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and augmentation settings for binary semantic segmentation."""

    image_height: int = 512
    image_width: int = 512
    landscape_class_id: int = 7100
    clutter_class_id: int = 550
    ignore_index: int = 255
    val_fraction: float = 0.15
    seed: int = 42
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    shift_scale_rotate_prob: float = 0.4
    random_brightness_contrast_prob: float = 0.3
    gaussian_noise_prob: float = 0.1
    gaussian_blur_prob: float = 0.1
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelConfig:
    """Model topology and encoder settings."""

    encoder_name: str = "resnet50"
    pretrained_encoder: bool = True
    decoder_dropout: float = 0.10
    output_channels: int = 1
    freeze_encoder_at_start: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and checkpointing settings."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    eval_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    threshold: float = 0.5
    gradient_clip_norm: float = 1.0
    freeze_encoder_epochs: int = 0
    resume_from_checkpoint: Path | None = None
    checkpoint_name: str = "best_model.pt"
    last_checkpoint_name: str = "last_model.pt"
    seed: int = 42


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware-aware toggles for performance and determinism."""

    device: str = "auto"
    use_amp: bool = True
    use_channels_last: bool = True
    use_torch_compile: bool = False
    deterministic: bool = False
    allow_tf32: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for the full experiment."""

    project_name: str = "clutter_vs_landscape_isolator"
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    hardware: HardwareConfig = HardwareConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the experiment configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        """Reconstruct an experiment configuration from a saved dictionary."""

        return cls(
            project_name=str(data.get("project_name", "clutter_vs_landscape_isolator")),
            paths=PathsConfig(**dict(data.get("paths", {}))),
            dataset=DatasetConfig(**dict(data.get("dataset", {}))),
            model=ModelConfig(**dict(data.get("model", {}))),
            training=TrainingConfig(**dict(data.get("training", {}))),
            hardware=HardwareConfig(**dict(data.get("hardware", {}))),
        )


def build_default_config() -> ExperimentConfig:
    """Create the default experiment configuration used by the CLI tools."""

    return ExperimentConfig()
