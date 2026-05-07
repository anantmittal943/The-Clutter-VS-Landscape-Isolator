# The Clutter vs. Landscape Isolator

This project trains a high-capacity binary semantic segmentation model to separate navigable **Landscape** from **Ground Clutter** in synthetic terrain imagery.

## Project layout

- `config.py` centralizes dataset paths, label IDs, model settings, training hyperparameters, and hardware toggles.
- `dataset.py` provides a robust PyTorch dataset, pair discovery, safe file loading, and Albumentations-based augmentation.
- `model.py` defines a ResNet-50 encoder U-Net with a clean encoder freeze/unfreeze interface.
- `train.py` runs the full training loop with structured logging, AMP, and best-checkpoint saving by validation mIoU.
- `test.py` runs timed inference, reports mean IoU, and prints a confusion matrix.
- `utils.py` contains reproducibility, metric, runtime, and checkpoint helpers.

## Environment setup

1. Create a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

The code targets PyTorch, torchvision, Albumentations, OpenCV, Pillow, and NumPy.

## Dataset format

Use paired image and mask folders with matching file stems:

```text
data/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Masks must be single-channel label maps. The default label mapping is:

- `550` -> Ground Clutter -> class `0`
- `7100` -> Landscape -> class `1`
- any other label -> ignored with index `255`

If your validation or test split is not provided explicitly, the training pipeline can split the training set using the configured validation fraction.

## Training

Run the default training pipeline:

```bash
python train.py
```

Override common settings when needed:

```bash
python train.py --train-image-dir data/train/images --train-mask-dir data/train/masks --val-image-dir data/val/images --val-mask-dir data/val/masks --epochs 80 --batch-size 4 --learning-rate 1e-4 --device cuda
```

Training uses:

- binary BCE + Dice loss with ignore-mask support
- AMP when CUDA is available
- best-checkpoint selection by validation mean IoU
- a last-checkpoint snapshot each epoch

## Inference and benchmarking

Evaluate the best checkpoint on the test split and measure image-level inference latency:

```bash
python test.py --checkpoint artifacts/checkpoints/best_model.pt --device cuda --split test
```

The benchmark measures per-image model latency with `time.perf_counter()` after a short warm-up. The timing excludes file I/O and dataset decoding so the result reflects model-side inference cost.

## Performance notes

The implementation is optimized for practical GPU usage:

- AMP is enabled by default on CUDA.
- Channel-last memory format is enabled by default for convolution-heavy workloads.
- The runtime exposes an opt-in toggle for `torch.compile` if your environment supports it.
- The inference script uses batch size 1 timing, which is the most relevant case for latency-sensitive deployment.

If you need to push latency below a target such as 50 ms/image, the first levers to test are input resolution, encoder depth, and whether `torch.compile` is supported in your runtime.

## Debugging misclassifications

For this project, a useful failure-analysis workflow is to inspect false positives and false negatives by texture family. Common bins include coarse gravel, dark soil, debris shadows, tire tracks, and synthetic edge artifacts. Tracking these groups makes it easier to decide whether the next improvement should come from augmentation, sampling, or architecture changes.
