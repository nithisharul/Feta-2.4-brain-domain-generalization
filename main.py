"""
main.py
-------
Training entry point for the fetal brain domain-generalisation pipeline.

Usage::

    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --resume outputs/checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.dataloader import load_config, get_dataloaders
from src.preprocessing.transforms import PreprocessingPipeline, TrainingPipeline
from src.models.unet3d import build_unet3d
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, get_device, format_params, count_parameters, save_json


def main(args: argparse.Namespace) -> None:
    # ---- Config ----
    config = load_config(args.config)
    sys_cfg = config["system"]

    set_seed(sys_cfg.get("seed", 42))
    device = get_device(sys_cfg.get("device", "cpu"))
    print(f"Device: {device}")

    # ---- Preprocessing ----
    pre_cfg = config.get("preprocessing", {})
    crop_size = tuple(pre_cfg.get("crop_size", [128, 128, 128]))
    clip_low = pre_cfg.get("clip_percentile_low", 0.5)
    clip_high = pre_cfg.get("clip_percentile_high", 99.5)

    transform_train = TrainingPipeline(crop_size=crop_size, clip_low=clip_low, clip_high=clip_high)
    transform_val = PreprocessingPipeline(crop_size=crop_size, clip_low=clip_low, clip_high=clip_high)

    # ---- DataLoaders ----
    train_loader, val_loader = get_dataloaders(
        config,
        transform_train=transform_train,
        transform_val=transform_val,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = build_unet3d(config)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({format_params(n_params)})")

    # ---- Trainer ----
    if args.resume:
        trainer = Trainer.load_checkpoint(args.resume, model, config, device=device)
    else:
        trainer = Trainer(model, config, device=device)

    # ---- Train ----
    history = trainer.train(train_loader, val_loader)

    # ---- Save history ----
    output_dir = Path(sys_cfg.get("output_dir", "outputs"))
    save_json(history, output_dir / "training_history.json")
    print(f"Training history saved to {output_dir / 'training_history.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3D U-Net for fetal brain MRI segmentation."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pth file to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())