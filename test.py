"""Inference and benchmarking entry point for the segmentation model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.cuda.amp import autocast

from config import ExperimentConfig, build_default_config
from dataset import build_dataloaders
from model import build_model
from utils import (
    compute_binary_metrics,
    configure_torch_runtime,
    empty_confusion_matrix,
    format_confusion_matrix,
    load_checkpoint,
    predict_binary_mask,
    resolve_device,
    update_confusion_matrix,
)

LOGGER = logging.getLogger("test")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark and evaluation script."""

    parser = argparse.ArgumentParser(
        description="Evaluate the trained binary segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", type=str, choices=("validation", "test"), default="test")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=3)
    return parser.parse_args()


def setup_logging() -> None:
    """Configure standard logging for evaluation output."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def evaluate_checkpoint(config: ExperimentConfig, checkpoint_path: Path, split: str, device: torch.device, use_amp: bool, threshold: float, warmup_steps: int) -> None:
    """Load a checkpoint, run inference, and report IoU plus timing statistics."""

    loaders = build_dataloaders(config)
    loader = loaders.validation if split == "validation" else loaders.test

    model = build_model(config.model).to(device)
    if config.hardware.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion_matrix = empty_confusion_matrix()
    timings: list[float] = []
    sample_index = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if config.hardware.use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            for image_index in range(images.size(0)):
                image = images[image_index : image_index + 1]
                mask = masks[image_index : image_index + 1]

                if sample_index < warmup_steps:
                    with autocast(enabled=use_amp):
                        _ = model(image)
                    sample_index += 1
                    continue

                start_time = perf_counter()
                with autocast(enabled=use_amp):
                    logits = model(image)
                elapsed = perf_counter() - start_time
                timings.append(elapsed)

                predictions = predict_binary_mask(logits, threshold)
                update_confusion_matrix(confusion_matrix, predictions, mask, config.dataset.ignore_index)
                sample_index += 1

    metrics = compute_binary_metrics(confusion_matrix)
    mean_latency_ms = (sum(timings) / max(len(timings), 1)) * 1000.0
    throughput = 1.0 / (sum(timings) / max(len(timings), 1)) if timings else 0.0

    LOGGER.info("Evaluation split: %s", split)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Processed images: %d", len(timings))
    LOGGER.info("Mean latency per image: %.3f ms", mean_latency_ms)
    LOGGER.info("Throughput: %.2f images/s", throughput)
    LOGGER.info("Metrics: %s", metrics)
    LOGGER.info("%s", format_confusion_matrix(confusion_matrix))


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    setup_logging()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint.get("config", build_default_config().to_dict()))
    if args.device is not None:
        config = ExperimentConfig.from_dict({**config.to_dict(), "hardware": {**config.hardware.__dict__, "device": args.device}})
    if args.threshold is not None:
        config = ExperimentConfig.from_dict({**config.to_dict(), "training": {**config.training.__dict__, "threshold": args.threshold}})

    device = resolve_device(config.hardware.device)
    configure_torch_runtime(allow_tf32=config.hardware.allow_tf32, deterministic=config.hardware.deterministic)
    evaluate_checkpoint(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        device=device,
        use_amp=config.hardware.use_amp and not args.no_amp and device.type == "cuda",
        threshold=config.training.threshold,
        warmup_steps=max(0, args.warmup_steps),
    )


if __name__ == "__main__":
    main()
