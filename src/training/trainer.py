"""
trainer.py
----------
Training loop with Dice + Cross-Entropy loss, LR scheduling,
checkpointing, and validation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Converts logits to probabilities via softmax, then computes per-class
    Dice and averages (macro).  The background class (0) is **included** by
    default so the model is penalised for false-positive background
    predictions in foreground regions.

    Args:
        num_classes: Total number of segmentation classes.
        smooth:      Laplace smoothing constant to avoid division by zero.
        ignore_background: If True, class 0 is excluded from the mean.
    """

    def __init__(
        self,
        num_classes: int = 7,
        smooth: float = 1e-5,
        ignore_background: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, D, H, W]  raw network output.
            targets: [B, D, H, W]     integer class labels.

        Returns:
            Scalar loss value.
        """
        probs = F.softmax(logits, dim=1)                   # [B, C, D, H, W]
        one_hot = F.one_hot(targets, self.num_classes)     # [B, D, H, W, C]
        one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        # Flatten spatial dims
        probs_flat = probs.view(probs.shape[0], self.num_classes, -1)
        oh_flat = one_hot.view(one_hot.shape[0], self.num_classes, -1)

        intersection = (probs_flat * oh_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + oh_flat.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.ignore_background:
            dice_per_class = dice_per_class[:, 1:]

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of Dice loss and Cross-Entropy loss.

    Args:
        num_classes:     Number of segmentation classes.
        dice_weight:     Weight for the Dice loss term.
        ce_weight:       Weight for the CE loss term.
        ignore_background: Exclude class 0 from Dice computation.
    """

    def __init__(
        self,
        num_classes: int = 7,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes, ignore_background=ignore_background)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (total_loss, dice_loss, ce_loss) — individual terms for logging.
        """
        d = self.dice(logits, targets)
        c = self.ce(logits, targets)
        total = self.dice_weight * d + self.ce_weight * c
        return total, d, c


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training and validation loop for the 3-D U-Net segmentation model.

    Features:
    - Combined Dice + CE loss
    - Adam optimiser with cosine / step / none LR schedule
    - Per-epoch validation with mean Dice score
    - Best-model checkpoint + periodic checkpoints
    - Early stopping

    Args:
        model:   The segmentation model.
        config:  Loaded YAML config dict.
        device:  Override the device from config (useful for testing).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> None:
        train_cfg = config["training"]
        sys_cfg = config["system"]

        self.config = config
        self.device = device or torch.device(sys_cfg.get("device", "cpu"))
        self.model = model.to(self.device)

        # Loss
        self.criterion = CombinedLoss(
            num_classes=config["data"]["num_classes"],
            dice_weight=train_cfg.get("dice_weight", 0.5),
            ce_weight=train_cfg.get("ce_weight", 0.5),
        )

        # Optimiser
        self.optimizer = Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # LR scheduler
        sched = train_cfg.get("scheduler", "cosine")
        if sched == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get("scheduler_T_max", train_cfg["num_epochs"]),
            )
        elif sched == "step":
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None

        # Hyperparameters
        self.num_epochs = train_cfg["num_epochs"]
        self.save_every = train_cfg.get("save_every", 10)
        self.log_interval = sys_cfg.get("log_interval", 10)
        self.patience = train_cfg.get("early_stopping_patience", 20)

        # Output paths
        self.output_dir = Path(sys_cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.best_val_dice: float = 0.0
        self.epochs_without_improvement: int = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_dice": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Args:
            train_loader: DataLoader for training set.
            val_loader:   DataLoader for validation set (optional).

        Returns:
            History dict with ``train_loss``, ``val_dice``, ``lr`` lists.
        """
        print(f"\n{'='*60}")
        print(f"  Training on {self.device} for {self.num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_loss = self._train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            # --- Validate ---
            val_dice = 0.0
            if val_loader is not None:
                val_dice = self._validate_epoch(val_loader)
                self.history["val_dice"].append(val_dice)

            # --- LR step ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:>4d}/{self.num_epochs}]  "
                f"loss={train_loss:.4f}  "
                f"val_dice={val_dice:.4f}  "
                f"lr={current_lr:.2e}  "
                f"({elapsed:.1f}s)"
            )

            # --- Checkpoints ---
            if val_loader is not None:
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, tag="best")
                    print(f"  ✓ New best val_dice={val_dice:.4f} — saved checkpoint.")
                else:
                    self.epochs_without_improvement += 1

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch:04d}")

            # --- Early stopping ---
            if (
                val_loader is not None
                and self.epochs_without_improvement >= self.patience
            ):
                print(
                    f"\n  Early stopping triggered after {self.patience} epochs "
                    f"without improvement."
                )
                break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val Dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}\n")
        return self.history

    def train_one_step(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        Run a single gradient update step.  Used by the sanity check.

        Returns:
            Scalar loss value as a Python float.
        """
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(images)
        loss, _, _ = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss, dice_l, ce_l = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            running_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                print(
                    f"  [ep {epoch}] batch {batch_idx}/{len(loader)}  "
                    f"loss={loss.item():.4f}  "
                    f"dice={dice_l.item():.4f}  "
                    f"ce={ce_l.item():.4f}"
                )

        return running_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> float:
        from ..evaluation.metrics import compute_multiclass_dice

        self.model.eval()
        dice_scores = []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            preds = logits.argmax(dim=1)
            dice = compute_multiclass_dice(
                preds, labels, num_classes=self.config["data"]["num_classes"]
            )
            dice_scores.append(dice["mean_dice"])

        return float(sum(dice_scores) / len(dice_scores))

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        fname = f"checkpoint_{tag}.pth"
        path = self.output_dir / fname
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_dice": self.best_val_dice,
                "history": self.history,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> "Trainer":
        """
        Restore a Trainer (model weights + optimiser state) from a checkpoint.

        Args:
            checkpoint_path: Path to the saved ``.pth`` file.
            model:           Model instance to load weights into.
            config:          Config dict.
            device:          Target device.

        Returns:
            A ``Trainer`` ready to resume training.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = cls(model, config, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.best_val_dice = ckpt.get("best_val_dice", 0.0)
        trainer.history = ckpt.get("history", trainer.history)
        print(f"Checkpoint loaded from {checkpoint_path} (epoch {ckpt['epoch']})")
        return trainer