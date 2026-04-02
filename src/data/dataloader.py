"""
dataloader.py
-------------
NIfTI loading, Dataset class, and DataLoader factory for the
fetal brain MRI segmentation pipeline.

Tissue label mapping (FeTA 2.4):
  0 - Background
  1 - External CSF
  2 - Gray Matter
  3 - White Matter
  4 - Ventricles
  5 - Cerebellum
  6 - Deep Gray Matter
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Label metadata
# ---------------------------------------------------------------------------

TISSUE_LABELS: Dict[int, str] = {
    0: "Background",
    1: "External CSF",
    2: "Gray Matter",
    3: "White Matter",
    4: "Ventricles",
    5: "Cerebellum",
    6: "Deep Gray Matter",
    7: "Brainstem",
}

NUM_CLASSES: int = len(TISSUE_LABELS)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load a YAML config file and return it as a plain dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FetalBrainDataset(Dataset):
    """
    PyTorch Dataset for 3D fetal brain MRI NIfTI volumes.

    Expected directory layout::

        images_dir/
            sub-001_T2w.nii.gz
            sub-002_T2w.nii.gz
            ...
        labels_dir/
            sub-001_T2w_dseg.nii.gz
            sub-002_T2w_dseg.nii.gz
            ...

    The dataset matches images to labels by stripping the common suffix
    ``_dseg`` from the label filename so that both share the same stem.
    If your naming convention differs, override ``_build_pairs``.

    Args:
        images_dir: Path to directory containing image NIfTI files.
        labels_dir: Path to directory containing label NIfTI files.
        transform: Optional callable applied to ``(image, label)`` pair.
            Should accept ``(np.ndarray [D,H,W], np.ndarray [D,H,W])``
            and return a ``(torch.Tensor [1,D,H,W], torch.Tensor [D,H,W])``.
        file_extension: File extension used to filter NIfTI volumes.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
        file_extension: str = ".nii.gz",
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.file_extension = file_extension

        self.pairs: List[Tuple[Path, Path]] = self._build_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image/label pairs found.\n"
                f"  images_dir : {self.images_dir}\n"
                f"  labels_dir : {self.labels_dir}\n"
                f"  extension  : {self.file_extension}\n"
                "Check that the directories exist and filenames match."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Match image files to label files, handling FeTA 2.4 naming conventions.

        FeTA 2.4 uses BIDS-style naming:
          Image : sub-001_rec-mial_T2w.nii.gz
          Label : sub-001_rec-mial_dseg.nii.gz   ← no _T2w in label name

        Strategy:
          1. Strip extension + MRI-suffix tokens (_T2w, _t2w, _rec-mial, etc.)
             from each image filename to get a subject key.
          2. Strip extension + _dseg from each label filename to get a subject key.
          3. Match on subject key.
        """
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        ext = self.file_extension

        # Tokens to strip when computing the subject key
        _STRIP = ["_T2w", "_t2w", "_rec-mial", "_rec-irtk", "_dseg"]

        def _subject_key(filename: str) -> str:
            key = filename.replace(ext, "")
            for tok in _STRIP:
                key = key.replace(tok, "")
            return key

        image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.name.endswith(ext)]
        )

        # Build subject_key → label_path map
        label_map: Dict[str, Path] = {}
        for p in self.labels_dir.iterdir():
            if p.name.endswith(ext):
                label_map[_subject_key(p.name)] = p

        pairs: List[Tuple[Path, Path]] = []
        unmatched: List[str] = []
        for img_path in image_files:
            key = _subject_key(img_path.name)
            label_path = label_map.get(key)
            if label_path is not None:
                pairs.append((img_path, label_path))
            else:
                unmatched.append(img_path.name)

        if unmatched:
            import warnings
            warnings.warn(
                f"Could not find labels for {len(unmatched)} image(s):\n"
                + "\n".join(f"  {n}" for n in unmatched[:10])
                + ("\n  ..." if len(unmatched) > 10 else ""),
                UserWarning,
                stacklevel=3,
            )

        return pairs

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        """Load a NIfTI volume and return it as a float32 numpy array."""
        vol = nib.load(str(path))
        arr = vol.get_fdata(dtype=np.float32)
        # Ensure 3-D (drop singleton trailing dims if present)
        arr = arr.squeeze()
        if arr.ndim != 3:
            raise ValueError(
                f"Expected a 3-D volume, got shape {arr.shape} from {path}"
            )
        return arr

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.pairs[idx]

        image = self._load_nifti(img_path)          # [D, H, W]  float32
        label = self._load_nifti(lbl_path)          # [D, H, W]  float32 → int

        label = label.astype(np.int64)
        # Safety clamp — ensures no label value exceeds num_classes-1
        label = np.clip(label, 0, NUM_CLASSES - 1)

        if self.transform is not None:
            image, label = self.transform(image, label)
        else:
            # Minimal conversion: add channel dim and cast
            image = torch.from_numpy(image).unsqueeze(0)   # [1, D, H, W]
            label = torch.from_numpy(label)                # [D, H, W]

        return image, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_subject_id(self, idx: int) -> str:
        """Return the image filename stem for the given index."""
        return self.pairs[idx][0].name.replace(self.file_extension, "")


# ---------------------------------------------------------------------------
# Config-driven Dataset wrapper
# ---------------------------------------------------------------------------

class FetalBrainDatasetFromConfig(FetalBrainDataset):
    """
    Convenience subclass that reads ``images_dir``, ``labels_dir``,
    and ``file_extension`` directly from a loaded config dict.
    """

    def __init__(self, config: Dict, transform: Optional[Callable] = None) -> None:
        data_cfg = config["data"]
        super().__init__(
            images_dir=data_cfg["images_dir"],
            labels_dir=data_cfg["labels_dir"],
            transform=transform,
            file_extension=data_cfg.get("file_extension", ".nii.gz"),
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    config: Dict,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train / validation DataLoaders from a config dict.

    The full dataset is split into train/val subsets according to
    ``config['data']['val_split']`` using a reproducible random shuffle
    seeded by ``config['system']['seed']``.

    Args:
        config: Loaded YAML config dict.
        transform_train: Transform applied to training samples.
        transform_val: Transform applied to validation samples.

    Returns:
        ``(train_loader, val_loader)``
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    sys_cfg = config["system"]

    full_dataset = FetalBrainDatasetFromConfig(config, transform=None)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * data_cfg.get("val_split", 0.2)))
    n_train = n_total - n_val

    # Reproducible split
    rng = random.Random(sys_cfg.get("seed", 42))
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = _SubsetWithTransform(full_dataset, train_indices, transform_train)
    val_subset = _SubsetWithTransform(full_dataset, val_indices, transform_val)

    num_workers = sys_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_subset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(sys_cfg.get("device", "cpu") != "cpu"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(sys_cfg.get("device", "cpu") != "cpu"),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Internal helper: subset with per-split transform
# ---------------------------------------------------------------------------

class _SubsetWithTransform(Dataset):
    """Apply a transform to a subset of another dataset."""

    def __init__(
        self,
        dataset: FetalBrainDataset,
        indices: List[int],
        transform: Optional[Callable],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img_path, lbl_path = self.dataset.pairs[real_idx]

        image = self.dataset._load_nifti(img_path)
        label = self.dataset._load_nifti(lbl_path).astype(np.int64)
        label = np.clip(label, 0, NUM_CLASSES - 1)

        if self.transform is not None:
            image, label = self.transform(image, label)
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            label = torch.from_numpy(label)

        return image, label