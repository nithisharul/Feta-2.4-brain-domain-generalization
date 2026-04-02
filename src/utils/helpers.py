"""
helpers.py
----------
Utility functions: seeding, device selection, parameter counting, I/O.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Also enables deterministic cuDNN behaviour at a small performance cost.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    If ``device_str`` is ``None`` or ``"auto"``, CUDA is used when
    available, otherwise CPU.

    Args:
        device_str: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``None``/``"auto"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load and return the contents of a JSON file."""
    with open(Path(path), "r") as f:
        return json.load(f)


def format_params(n: int) -> str:
    """Return a human-readable parameter count string, e.g. ``'26.3M'``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)