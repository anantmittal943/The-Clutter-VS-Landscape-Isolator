"""Training entry point for the Clutter vs. Landscape isolator."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import AdamW

from config import ExperimentConfig, build_default_config
from dataset import DataLoaders, build_dataloaders
from model import build_model
from utils import (
    configure_torch_runtime,
    compute_binary_metrics,
    empty_confusion_matrix,
    ensure_directory,
    format_confusion_matrix,
    load_checkpoint,
    predict_binary_mask,
    save_checkpoint,
    seed_everything,
    update_confusion_matrix,
    resolve_device,
)

LOGGER = logging.getLogger("train")


class MaskedBCEDiceLoss(nn.Module):
    """Binary cross-entropy plus Dice loss with ignore-index masking."""

    def __init__(self, bce_weight: float, dice_weight: float, ignore_index: int) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        logits_flat = logits.squeeze(1)[valid_mask].float()
        targets_flat = targets[valid_mask].float()

        bce_loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat)
        probabilities = torch.sigmoid(logits.squeeze(1))[valid_mask].float()
        intersection = torch.sum(probabilities * targets_flat)
        denominator = torch.sum(probabilities) + torch.sum(targets_flat)
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training process."""

    parser = argparse.ArgumentParser(
        description="Train a high-capacity U-Net to isolate navigable landscape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-mask-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--val-mask-dir", type=Path, default=None)
    parser.add_argument("--test-image-dir", type=Path, default=None)
    parser.add_argument("--test-mask-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def setup_logging(log_dir: Path) -> None:
    """Configure standard logging for the training run."""

    ensure_directory(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply CLI overrides to the immutable configuration object."""

    paths = config.paths
    dataset = config.dataset
    training = config.training
    hardware = config.hardware

    if args.train_image_dir is not None:
        paths = replace(paths, train_image_dir=args.train_image_dir)
    if args.train_mask_dir is not None:
        paths = replace(paths, train_mask_dir=args.train_mask_dir)
    if args.val_image_dir is not None:
        paths = replace(paths, val_image_dir=args.val_image_dir)
    if args.val_mask_dir is not None:
        paths = replace(paths, val_mask_dir=args.val_mask_dir)
    if args.test_image_dir is not None:
        paths = replace(paths, test_image_dir=args.test_image_dir)
    if args.test_mask_dir is not None:
        paths = replace(paths, test_mask_dir=args.test_mask_dir)
    if args.output_dir is not None:
        paths = replace(
            paths,
            output_dir=args.output_dir,
            checkpoint_dir=args.output_dir / "checkpoints",
            log_dir=args.output_dir / "logs",
            predictions_dir=args.output_dir / "predictions",
        )
    if args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    if args.learning_rate is not None:
        training = replace(training, learning_rate=args.learning_rate)
    if args.weight_decay is not None:
        training = replace(training, weight_decay=args.weight_decay)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    if args.eval_batch_size is not None:
        training = replace(training, eval_batch_size=args.eval_batch_size)
    if args.num_workers is not None:
        training = replace(training, num_workers=args.num_workers)
    if args.resume is not None:
        training = replace(training, resume_from_checkpoint=args.resume)
    if args.seed is not None:
        training = replace(training, seed=args.seed)
    if args.device is not None:
        hardware = replace(hardware, device=args.device)
    if args.no_amp:
        hardware = replace(hardware, use_amp=False)
    if args.no_compile:
        hardware = replace(hardware, use_torch_compile=False)

    return replace(config, paths=paths, dataset=dataset, training=training, hardware=hardware)


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    use_channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to the target device while preserving segmentation layout."""

    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    if use_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, masks


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MaskedBCEDiceLoss,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    use_channels_last: bool,
    threshold: float,
    ignore_index: int,
    max_grad_norm: float,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Run one training epoch and return loss and metrics."""

    model.train()
    running_loss = 0.0
    sample_count = 0
    confusion_matrix = empty_confusion_matrix()

    for batch in loader:
        images, masks = move_batch_to_device(batch, device, use_channels_last)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        predictions = predict_binary_mask(logits.detach(), threshold)
        update_confusion_matrix(confusion_matrix, predictions, masks, ignore_index)

    metrics = compute_binary_metrics(confusion_matrix)
    average_loss = running_loss / max(sample_count, 1)
    return average_loss, metrics, confusion_matrix


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedBCEDiceLoss,
    device: torch.device,
    use_amp: bool,
    use_channels_last: bool,
    threshold: float,
    ignore_index: int,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Evaluate the model on validation or test data."""

    model.eval()
    running_loss = 0.0
    sample_count = 0
    confusion_matrix = empty_confusion_matrix()

    for batch in loader:
        images, masks = move_batch_to_device(batch, device, use_channels_last)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        predictions = predict_binary_mask(logits, threshold)
        update_confusion_matrix(confusion_matrix, predictions, masks, ignore_index)

    metrics = compute_binary_metrics(confusion_matrix)
    average_loss = running_loss / max(sample_count, 1)
    return average_loss, metrics, confusion_matrix


def train(config: ExperimentConfig) -> None:
    """Execute the end-to-end training pipeline."""

    setup_logging(config.paths.log_dir)
    seed_everything(config.training.seed)
    configure_torch_runtime(
        allow_tf32=config.hardware.allow_tf32,
        deterministic=config.hardware.deterministic,
    )

    device = resolve_device(config.hardware.device)
    LOGGER.info("Using device: %s", device)
    loaders: DataLoaders = build_dataloaders(config)

    base_model = build_model(config.model).to(device)
    if config.hardware.use_channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    if config.model.freeze_encoder_at_start and config.training.freeze_encoder_epochs > 0:
        base_model.freeze_encoder()
    if config.hardware.use_torch_compile and hasattr(torch, "compile"):
        # Torch compile can reduce inference/training overhead on supported setups,
        # but it is intentionally opt-in because some environments are not compatible.
        model = torch.compile(base_model)  # type: ignore[assignment]
    else:
        model = base_model

    optimizer = AdamW(
        base_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = MaskedBCEDiceLoss(
        bce_weight=config.training.bce_weight,
        dice_weight=config.training.dice_weight,
        ignore_index=config.dataset.ignore_index,
    )
    scaler = GradScaler(enabled=config.hardware.use_amp and device.type == "cuda")

    start_epoch = 1
    best_val_iou = -1.0
    if config.training.resume_from_checkpoint is not None:
        checkpoint = load_checkpoint(config.training.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_iou = float(checkpoint.get("best_val_iou", -1.0))
        LOGGER.info(
            "Resumed from %s at epoch %d with best val IoU %.4f",
            config.training.resume_from_checkpoint,
            start_epoch,
            best_val_iou,
        )

    train_loader = loaders.train
    validation_loader = loaders.validation
    best_checkpoint_path = config.paths.checkpoint_dir / config.training.checkpoint_name
    last_checkpoint_path = config.paths.checkpoint_dir / config.training.last_checkpoint_name

    for epoch in range(start_epoch, config.training.epochs + 1):
        if config.training.freeze_encoder_epochs > 0 and epoch <= config.training.freeze_encoder_epochs:
            base_model.freeze_encoder()
        else:
            base_model.unfreeze_encoder()

        train_loss, train_metrics, train_confusion = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=config.hardware.use_amp and device.type == "cuda",
            use_channels_last=config.hardware.use_channels_last,
            threshold=config.training.threshold,
            ignore_index=config.dataset.ignore_index,
            max_grad_norm=config.training.gradient_clip_norm,
        )
        val_loss, val_metrics, val_confusion = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=config.hardware.use_amp and device.type == "cuda",
            use_channels_last=config.hardware.use_channels_last,
            threshold=config.training.threshold,
            ignore_index=config.dataset.ignore_index,
        )

        LOGGER.info(
            "Epoch %03d/%03d | train_loss=%.4f | train_mIoU=%.4f | val_loss=%.4f | val_mIoU=%.4f",
            epoch,
            config.training.epochs,
            train_loss,
            train_metrics["mean_iou"],
            val_loss,
            val_metrics["mean_iou"],
        )

        LOGGER.info("Train metrics: %s", train_metrics)
        LOGGER.info("Validation metrics: %s", val_metrics)

        save_checkpoint(
            last_checkpoint_path,
            {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val_iou": best_val_iou,
                "config": config.to_dict(),
                "train_confusion_matrix": train_confusion,
                "val_confusion_matrix": val_confusion,
            },
        )

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            save_checkpoint(
                best_checkpoint_path,
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_val_iou": best_val_iou,
                    "config": config.to_dict(),
                    "train_confusion_matrix": train_confusion,
                    "val_confusion_matrix": val_confusion,
                },
            )
            LOGGER.info("Saved new best checkpoint to %s (val mIoU=%.4f)", best_checkpoint_path, best_val_iou)

    LOGGER.info("Training complete. Best validation mIoU: %.4f", best_val_iou)
    LOGGER.info(format_confusion_matrix(val_confusion))


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = apply_overrides(build_default_config(), args)
    train(config)


if __name__ == "__main__":
    main()
