"""Dataset and DataLoader utilities for the binary segmentation task."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import DatasetConfig, ExperimentConfig
from utils import ensure_directory

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationPair:
    """Image and mask path pair used by the dataset."""

    image_path: Path
    mask_path: Path


class BinarySegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    """PyTorch dataset that safely loads image-mask pairs and remaps labels."""

    def __init__(
        self,
        pairs: Sequence[SegmentationPair],
        dataset_config: DatasetConfig,
        transforms: A.Compose | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.dataset_config = dataset_config
        self.transforms = transforms
        self._failed_pairs: set[tuple[Path, Path]] = set()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if not self.pairs:
            raise IndexError("The dataset is empty.")

        total_samples = len(self.pairs)
        for offset in range(total_samples):
            pair = self.pairs[(index + offset) % total_samples]
            try:
                return self._load_pair(pair)
            except Exception as exc:  # pragma: no cover - safety path
                if (pair.image_path, pair.mask_path) not in self._failed_pairs:
                    self._failed_pairs.add((pair.image_path, pair.mask_path))
                    LOGGER.warning(
                        "Skipping corrupted sample image=%s mask=%s error=%s",
                        pair.image_path,
                        pair.mask_path,
                        exc,
                    )
        raise RuntimeError("Unable to load any valid sample from the dataset.")

    def _load_pair(self, pair: SegmentationPair) -> dict[str, torch.Tensor | str]:
        image = self._safe_load_image(pair.image_path)
        mask = self._safe_load_mask(pair.mask_path)
        encoded_mask = self._encode_mask(mask)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=encoded_mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(encoded_mask)

        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.as_tensor(image_tensor)
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.as_tensor(mask_tensor)

        return {
            "image": image_tensor.float(),
            "mask": mask_tensor.long(),
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }

    def _safe_load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.load()
            return np.asarray(image, dtype=np.uint8)

    def _safe_load_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as mask:
            mask = mask.convert("I")
            mask.load()
            return np.asarray(mask, dtype=np.int64)

    def _encode_mask(self, mask: np.ndarray) -> np.ndarray:
        if np.isin(mask, [0, 1, self.dataset_config.ignore_index]).all():
            return mask.astype(np.uint8)

        encoded_mask = np.full(mask.shape, self.dataset_config.ignore_index, dtype=np.uint8)
        encoded_mask[mask == self.dataset_config.clutter_class_id] = 0
        encoded_mask[mask == self.dataset_config.landscape_class_id] = 1
        return encoded_mask


def discover_pairs(image_dir: Path, mask_dir: Path) -> list[SegmentationPair]:
    """Pair image and mask files by stem name inside two directories."""

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    mask_lookup: dict[str, Path] = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if mask_path.is_file() and mask_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            mask_lookup[mask_path.stem] = mask_path

    pairs: list[SegmentationPair] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        mask_path = mask_lookup.get(image_path.stem)
        if mask_path is None:
            LOGGER.warning("No matching mask found for image %s", image_path)
            continue
        pairs.append(SegmentationPair(image_path=image_path, mask_path=mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matched image-mask pairs found in {image_dir} and {mask_dir}."
        )
    return pairs


def split_pairs(
    pairs: Sequence[SegmentationPair],
    validation_fraction: float,
    seed: int,
) -> tuple[list[SegmentationPair], list[SegmentationPair]]:
    """Split pairs into train and validation subsets."""

    shuffled_pairs = list(pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    validation_size = max(1, int(len(shuffled_pairs) * validation_fraction))
    validation_pairs = shuffled_pairs[:validation_size]
    train_pairs = shuffled_pairs[validation_size:]
    if not train_pairs:
        raise ValueError("Training split is empty after applying the validation fraction.")
    return train_pairs, validation_pairs


def build_transforms(dataset_config: DatasetConfig, train: bool) -> A.Compose:
    """Create the Albumentations pipeline for training or evaluation."""

    resize = A.Resize(
        height=dataset_config.image_height,
        width=dataset_config.image_width,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )

    if train:
        return A.Compose(
            [
                resize,
                A.HorizontalFlip(p=dataset_config.horizontal_flip_prob),
                A.VerticalFlip(p=dataset_config.vertical_flip_prob),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=dataset_config.shift_scale_rotate_prob,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=dataset_config.random_brightness_contrast_prob,
                ),
                A.GaussNoise(p=dataset_config.gaussian_noise_prob),
                A.GaussianBlur(blur_limit=(3, 5), p=dataset_config.gaussian_blur_prob),
                A.Normalize(
                    mean=dataset_config.normalize_mean,
                    std=dataset_config.normalize_std,
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            resize,
            A.Normalize(
                mean=dataset_config.normalize_mean,
                std=dataset_config.normalize_std,
            ),
            ToTensorV2(),
        ]
    )


def _build_loader(
    dataset: BinarySegmentationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader[dict[str, torch.Tensor | str]]:
    """Build a DataLoader with sane defaults for segmentation workloads."""

    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": _collate_segmentation_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _collate_segmentation_batch(
    batch: Sequence[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate function that preserves image and mask tensors with metadata."""

    images = torch.stack([sample["image"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    image_paths = [str(sample["image_path"]) for sample in batch]
    mask_paths = [str(sample["mask_path"]) for sample in batch]
    return {
        "image": images,
        "mask": masks,
        "image_path": image_paths,
        "mask_path": mask_paths,
    }


@dataclass(frozen=True)
class DataLoaders:
    """Convenience wrapper for the three data splits used by the pipeline."""

    train: DataLoader[dict[str, torch.Tensor | list[str]]]
    validation: DataLoader[dict[str, torch.Tensor | list[str]]]
    test: DataLoader[dict[str, torch.Tensor | list[str]]]


def build_dataloaders(config: ExperimentConfig) -> DataLoaders:
    """Construct train, validation, and test loaders from the experiment config."""

    train_pairs = discover_pairs(config.paths.train_image_dir, config.paths.train_mask_dir)
    val_image_dir = config.paths.val_image_dir
    val_mask_dir = config.paths.val_mask_dir
    if val_image_dir.exists() and val_mask_dir.exists():
        validation_pairs = discover_pairs(val_image_dir, val_mask_dir)
    else:
        train_pairs, validation_pairs = split_pairs(
            train_pairs,
            validation_fraction=config.dataset.val_fraction,
            seed=config.dataset.seed,
        )

    test_pairs: list[SegmentationPair]
    if config.paths.test_image_dir.exists() and config.paths.test_mask_dir.exists():
        test_pairs = discover_pairs(config.paths.test_image_dir, config.paths.test_mask_dir)
    else:
        test_pairs = list(validation_pairs)

    train_dataset = BinarySegmentationDataset(
        pairs=train_pairs,
        dataset_config=config.dataset,
        transforms=build_transforms(config.dataset, train=True),
    )
    validation_dataset = BinarySegmentationDataset(
        pairs=validation_pairs,
        dataset_config=config.dataset,
        transforms=build_transforms(config.dataset, train=False),
    )
    test_dataset = BinarySegmentationDataset(
        pairs=test_pairs,
        dataset_config=config.dataset,
        transforms=build_transforms(config.dataset, train=False),
    )

    ensure_directory(config.paths.output_dir)
    ensure_directory(config.paths.checkpoint_dir)
    ensure_directory(config.paths.log_dir)
    ensure_directory(config.paths.predictions_dir)

    return DataLoaders(
        train=_build_loader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers,
        ),
        validation=_build_loader(
            validation_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers,
        ),
        test=_build_loader(
            test_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers,
        ),
    )
