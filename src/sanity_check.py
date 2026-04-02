"""
sanity_check.py
---------------
End-to-end pipeline verification script.

Runs all six stages of the pipeline on a single synthetic volume so
that the entire stack can be validated without real FeTA data.

Usage::

    python src/sanity_check.py --config configs/config.yaml
    python src/sanity_check.py --config configs/config.yaml --train

Expected output (abridged)::

    ============================================================
    FETAL BRAIN MRI SEGMENTATION - SANITY CHECK
    ============================================================

    [1/6] Loading configuration...
      ✓ Config loaded from: configs/config.yaml
    ...
    ============================================================
    SANITY CHECK PASSED ✓
    ============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure the project root is on sys.path when invoked directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import load_config
from src.preprocessing.transforms import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_multiclass_dice
from src.utils.helpers import set_seed, get_device, count_parameters, format_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_batch(
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create one fake MRI batch using the preprocessing pipeline.

    Returns:
        ``(images, labels)`` tensors of the correct shapes.
    """
    pre_cfg = config.get("preprocessing", {})
    crop_size = tuple(pre_cfg.get("crop_size", [128, 128, 128]))
    num_classes = config["data"]["num_classes"]

    pipeline = PreprocessingPipeline(crop_size=crop_size)

    # Random synthetic volume (D, H, W) with realistic MRI-like range
    vol = (np.random.randn(*crop_size) * 100 + 300).astype(np.float32)
    vol = np.clip(vol, 0, None)

    # Random label map
    lbl = np.random.randint(0, num_classes, size=crop_size).astype(np.int64)

    image_t, label_t = pipeline(vol, lbl)
    # Add batch dimension
    return image_t.unsqueeze(0), label_t.unsqueeze(0)


def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"\n  ✗ FAILED: {msg}")
        sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sanity_check(config_path: str, run_train: bool = False) -> None:
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  FETAL BRAIN MRI SEGMENTATION - SANITY CHECK")
    print(f"{SEP}\n")

    # ------------------------------------------------------------------
    # Stage 1 — Config
    # ------------------------------------------------------------------
    print("[1/6] Loading configuration...")
    config = load_config(config_path)
    _ok(f"Config loaded from: {config_path}")

    # Override device for fast sanity check
    config["system"]["device"] = "cpu"
    device = get_device("cpu")
    set_seed(config["system"].get("seed", 42))

    # ------------------------------------------------------------------
    # Stage 2 — Preprocessing
    # ------------------------------------------------------------------
    print("\n[2/6] Creating preprocessing transforms...")
    pre_cfg = config.get("preprocessing", {})
    crop_size = tuple(pre_cfg.get("crop_size", [128, 128, 128]))
    pipeline = PreprocessingPipeline(crop_size=crop_size)
    _ok("PreprocessingPipeline created")

    # ------------------------------------------------------------------
    # Stage 3 — Synthetic dataset
    # ------------------------------------------------------------------
    print("\n[3/6] Creating synthetic dataset...")
    images, labels = _make_synthetic_batch(config)
    _ok("Synthetic sample created with 1 sample")

    # ------------------------------------------------------------------
    # Stage 4 — DataLoader (from tensors)
    # ------------------------------------------------------------------
    print("\n[4/6] Creating dataloader...")
    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=1)
    _ok("DataLoader created")

    # ------------------------------------------------------------------
    # Stage 5 — Model
    # ------------------------------------------------------------------
    print("\n[5/6] Building model...")
    model = build_unet3d(config).to(device)
    n_params = count_parameters(model)
    _ok("3D U-Net model built")
    _ok(f"Total parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Stage 6 — Forward pass
    # ------------------------------------------------------------------
    print("\n[6/6] Running forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(images.to(device))

    print(f"  → Image tensor shape:  {list(images.shape)}")
    print(f"  → Label tensor shape:  {list(labels.shape)}")
    print(f"  → Model output shape:  {list(out.shape)}")

    expected_out = [1, config["data"]["num_classes"]] + list(crop_size)
    _check(list(out.shape) == expected_out, f"Output shape mismatch: {list(out.shape)} != {expected_out}")
    _ok("All shape checks passed!")

    # Dice sanity
    preds = out.argmax(dim=1)
    dice_results = compute_multiclass_dice(preds.cpu(), labels, num_classes=config["data"]["num_classes"])
    _ok(f"Metrics computed — mean Dice (random init): {dice_results['mean_dice']:.4f}")

    print(f"\n{SEP}")
    print("  SANITY CHECK PASSED ✓")
    print(f"{SEP}")

    # ------------------------------------------------------------------
    # Optional: one-step training test
    # ------------------------------------------------------------------
    if run_train:
        print(f"\n{SEP}")
        print("  ONE-STEP TRAINING TEST")
        print(f"{SEP}\n")

        trainer = Trainer(model, config, device=device)
        loss_val = trainer.train_one_step(images, labels)

        _check(np.isfinite(loss_val), f"Loss is not finite: {loss_val}")
        _ok(f"Loss value: {loss_val:.4f}")
        _ok("Gradients computed and weights updated!")

        print(f"\n{SEP}")
        print("  ONE-STEP TRAINING PASSED ✓")
        print(f"{SEP}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end sanity check for the fetal brain segmentation pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also run a one-step training test.",
    )
    args = parser.parse_args()
    run_sanity_check(args.config, run_train=args.train)


if __name__ == "__main__":
    main()