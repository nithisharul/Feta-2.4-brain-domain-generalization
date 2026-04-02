"""
metrics.py
----------
Multi-class Dice score computation for volumetric segmentation evaluation.

FeTA 2.4 tissue classes:
  0 - Background      3 - White Matter    6 - Deep Gray Matter
  1 - External CSF    4 - Ventricles
  2 - Gray Matter     5 - Cerebellum
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..data.dataloader import TISSUE_LABELS


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_multiclass_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 7,
    smooth: float = 1e-5,
    ignore_background: bool = False,
) -> Dict[str, float]:
    """
    Compute per-class and mean Dice scores for a batch of predictions.

    Args:
        preds:             [B, D, H, W] integer class predictions.
        targets:           [B, D, H, W] integer ground-truth labels.
        num_classes:       Total number of classes.
        smooth:            Laplace smoothing constant.
        ignore_background: If True, class 0 is excluded from mean Dice.

    Returns:
        Dictionary with keys:
        - ``"class_{i}"``  — Dice score for class *i*  (float 0..1)
        - ``"mean_dice"``  — Mean over all (or foreground-only) classes
        - ``"class_names"`` — Dict mapping class index to tissue name
    """
    preds = preds.long()
    targets = targets.long()

    results: Dict[str, float] = {}
    dice_values: List[float] = []

    start_class = 1 if ignore_background else 0

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        intersection = (pred_cls * tgt_cls).sum().item()
        denom = pred_cls.sum().item() + tgt_cls.sum().item()

        if denom == 0:
            # Class absent from both pred and target → perfect score
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (denom + smooth)

        results[f"class_{cls}"] = dice
        if cls >= start_class:
            dice_values.append(dice)

    results["mean_dice"] = float(sum(dice_values) / len(dice_values)) if dice_values else 0.0
    results["class_names"] = TISSUE_LABELS  # type: ignore[assignment]
    return results


# ---------------------------------------------------------------------------
# Stateful metric accumulator
# ---------------------------------------------------------------------------

class DiceMetric:
    """
    Stateful Dice metric accumulator.

    Useful when evaluating over a full dataset one batch at a time.
    Call :meth:`update` for each batch, then :meth:`compute` at the end.

    Example::

        metric = DiceMetric(num_classes=7)
        for images, labels in val_loader:
            preds = model(images).argmax(dim=1)
            metric.update(preds, labels)
        results = metric.compute()
        print(results["mean_dice"])

    Args:
        num_classes:       Total number of tissue classes.
        ignore_background: Exclude class 0 from mean Dice.
        smooth:            Smoothing constant.
    """

    def __init__(
        self,
        num_classes: int = 7,
        ignore_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        """Clear accumulated state."""
        self._intersection = [0.0] * self.num_classes
        self._denom = [0.0] * self.num_classes
        self._n_samples = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate statistics for one batch.

        Args:
            preds:   [B, D, H, W] integer predictions.
            targets: [B, D, H, W] integer ground-truth.
        """
        preds = preds.long()
        targets = targets.long()
        self._n_samples += preds.shape[0]
        for cls in range(self.num_classes):
            p = (preds == cls).float()
            t = (targets == cls).float()
            self._intersection[cls] += (p * t).sum().item()
            self._denom[cls] += (p.sum() + t.sum()).item()

    def compute(self) -> Dict[str, float]:
        """
        Return per-class and mean Dice over all accumulated samples.

        Returns:
            Same structure as :func:`compute_multiclass_dice`.
        """
        results: Dict[str, float] = {}
        dice_values: List[float] = []
        start_class = 1 if self.ignore_background else 0

        for cls in range(self.num_classes):
            d = self._denom[cls]
            if d == 0:
                dice = 1.0
            else:
                dice = (2.0 * self._intersection[cls] + self.smooth) / (d + self.smooth)
            results[f"class_{cls}"] = dice
            if cls >= start_class:
                dice_values.append(dice)

        results["mean_dice"] = float(sum(dice_values) / len(dice_values)) if dice_values else 0.0
        results["class_names"] = TISSUE_LABELS  # type: ignore[assignment]
        return results

    def pretty_print(self, results: Optional[Dict] = None) -> None:
        """Print a formatted table of Dice scores."""
        if results is None:
            results = self.compute()
        print("\n  Dice Scores")
        print("  " + "-" * 40)
        for cls in range(self.num_classes):
            name = TISSUE_LABELS.get(cls, f"Class {cls}")
            score = results.get(f"class_{cls}", float("nan"))
            print(f"  [{cls}] {name:<22s}  {score:.4f}")
        print("  " + "-" * 40)
        print(f"  Mean Dice:                 {results['mean_dice']:.4f}\n")