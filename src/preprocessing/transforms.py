"""
transforms.py
-------------
Composable preprocessing transforms for 3-D fetal brain MRI volumes.

All transforms accept (image: np.ndarray, label: np.ndarray) pairs where
both arrays have shape [D, H, W].  The final transform ``ToTensor3D``
converts them to PyTorch tensors and adds the channel dimension to the image.

Typical usage::

    pipeline = PreprocessingPipeline(crop_size=(128, 128, 128))
    image_tensor, label_tensor = pipeline(image_np, label_np)
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Transform:
    """Base class for all transforms."""

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class ClipPercentile(Transform):
    """
    Clip image intensities to the [low_pct, high_pct] percentile range.

    Operates only on the image; label is passed through unchanged.

    Args:
        low_pct:  Lower percentile (default 0.5).
        high_pct: Upper percentile (default 99.5).
    """

    def __init__(self, low_pct: float = 0.5, high_pct: float = 99.5) -> None:
        if not (0 <= low_pct < high_pct <= 100):
            raise ValueError(
                f"Percentiles must satisfy 0 <= low_pct < high_pct <= 100, "
                f"got ({low_pct}, {high_pct})."
            )
        self.low_pct = low_pct
        self.high_pct = high_pct

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lo = float(np.percentile(image, self.low_pct))
        hi = float(np.percentile(image, self.high_pct))
        image = np.clip(image, lo, hi)
        return image, label


class ZScoreNormalize(Transform):
    """
    Z-score normalise the image: ``(x - mean) / (std + eps)``.

    Stats are computed from the *foreground* voxels (intensity > 0) when
    ``foreground_only=True`` (default), which is standard practice for
    brain MRI where the background is roughly zero.

    Args:
        foreground_only: If True, compute stats only over non-zero voxels.
        eps: Small constant for numerical stability.
    """

    def __init__(self, foreground_only: bool = True, eps: float = 1e-8) -> None:
        self.foreground_only = foreground_only
        self.eps = eps

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.foreground_only:
            mask = image > 0
            if mask.sum() == 0:
                mask = np.ones_like(image, dtype=bool)
            mean = float(image[mask].mean())
            std = float(image[mask].std())
        else:
            mean = float(image.mean())
            std = float(image.std())

        image = (image - mean) / (std + self.eps)
        return image, label


class CenterCrop3D(Transform):
    """
    Crop (or pad) the volume to a fixed spatial size centred on the volume.

    The crop is performed symmetrically around the centre of each axis.
    If the volume is smaller than ``crop_size`` along any axis the volume
    is zero-padded on both sides.

    Args:
        crop_size: Target spatial size as ``(D, H, W)``.
    """

    def __init__(self, crop_size: Tuple[int, int, int]) -> None:
        if len(crop_size) != 3:
            raise ValueError(f"crop_size must have 3 elements, got {len(crop_size)}.")
        self.crop_size = tuple(crop_size)

    def _crop_or_pad(
        self, arr: np.ndarray, target: Tuple[int, int, int]
    ) -> np.ndarray:
        """Pad then crop a 3-D array to *target* shape."""
        result = arr
        for axis, tgt in enumerate(target):
            size = result.shape[axis]
            if size < tgt:
                pad_total = tgt - size
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode="constant", constant_values=0)

        # Now crop
        slices = []
        for axis, tgt in enumerate(target):
            size = result.shape[axis]
            start = (size - tgt) // 2
            slices.append(slice(start, start + tgt))
        return result[tuple(slices)]

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = self._crop_or_pad(image, self.crop_size)
        label = self._crop_or_pad(label, self.crop_size)
        return image, label


class ToTensor3D(Transform):
    """
    Convert numpy arrays to PyTorch tensors.

    * ``image`` [D, H, W] → ``torch.float32`` tensor [1, D, H, W]
    * ``label`` [D, H, W] → ``torch.int64``  tensor [D, H, W]
    """

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label_t = torch.from_numpy(label.astype(np.int64))
        return image_t, label_t


class RandomFlip3D(Transform):
    """
    Randomly flip the volume along each spatial axis independently.

    Useful as a light data-augmentation step. Applied to both image and label.

    Args:
        axes: Which axes to consider flipping (0=D, 1=H, 2=W).
        p:    Probability of flipping along each axis.
    """

    def __init__(
        self,
        axes: Tuple[int, ...] = (0, 1, 2),
        p: float = 0.5,
    ) -> None:
        self.axes = axes
        self.p = p

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for ax in self.axes:
            if np.random.random() < self.p:
                image = np.flip(image, axis=ax).copy()
                label = np.flip(label, axis=ax).copy()
        return image, label


class RandomIntensityShift(Transform):
    """
    Add a small random offset and scale to image intensities (augmentation).

    Applied **after** Z-score normalisation so the shift/scale values are
    in standardised units.

    Args:
        shift_range: Max absolute shift added to all voxels.
        scale_range: Multiplicative factor sampled uniformly from
                     ``[1 - scale_range, 1 + scale_range]``.
    """

    def __init__(self, shift_range: float = 0.1, scale_range: float = 0.1) -> None:
        self.shift_range = shift_range
        self.scale_range = scale_range

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        image = image * scale + shift
        return image, label


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class Compose:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PreprocessingPipeline(Compose):
    """
    Standard preprocessing pipeline used for both training and inference.

    Steps (in order):
    1. ``ClipPercentile``   — clip outlier intensities
    2. ``ZScoreNormalize``  — standardise intensity distribution
    3. ``CenterCrop3D``     — resize to ``crop_size``
    4. ``ToTensor3D``       — convert to PyTorch tensors

    Args:
        crop_size:          Target volume shape (D, H, W).
        clip_low:           Lower percentile for clipping.
        clip_high:          Upper percentile for clipping.
        foreground_only:    Use only non-zero voxels for Z-score stats.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        clip_low: float = 0.5,
        clip_high: float = 99.5,
        foreground_only: bool = True,
    ) -> None:
        self.crop_size = crop_size
        super().__init__(
            [
                ClipPercentile(clip_low, clip_high),
                ZScoreNormalize(foreground_only=foreground_only),
                CenterCrop3D(crop_size),
                ToTensor3D(),
            ]
        )

    @classmethod
    def from_config(cls, config: dict) -> "PreprocessingPipeline":
        """Instantiate directly from a loaded YAML config dict."""
        pre_cfg = config.get("preprocessing", {})
        crop_size = tuple(pre_cfg.get("crop_size", [128, 128, 128]))
        clip_low = pre_cfg.get("clip_percentile_low", 0.5)
        clip_high = pre_cfg.get("clip_percentile_high", 99.5)
        return cls(crop_size=crop_size, clip_low=clip_low, clip_high=clip_high)


class TrainingPipeline(Compose):
    """
    Preprocessing + augmentation pipeline for training only.

    Extends ``PreprocessingPipeline`` with random flips and intensity jitter
    inserted *before* tensor conversion.

    Args:
        crop_size:    Target volume shape.
        flip_p:       Probability of flipping along each axis.
        shift_range:  Intensity shift magnitude.
        scale_range:  Intensity scale jitter range.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        clip_low: float = 0.5,
        clip_high: float = 99.5,
        flip_p: float = 0.5,
        shift_range: float = 0.1,
        scale_range: float = 0.1,
    ) -> None:
        super().__init__(
            [
                ClipPercentile(clip_low, clip_high),
                ZScoreNormalize(foreground_only=True),
                CenterCrop3D(crop_size),
                RandomFlip3D(axes=(0, 1, 2), p=flip_p),
                RandomIntensityShift(shift_range, scale_range),
                ToTensor3D(),
            ]
        )