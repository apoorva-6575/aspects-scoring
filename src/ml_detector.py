"""Optional ML-assisted detector for Objective 1, meant to be OR-combined
with the classical symmetry-difference detector (detection.py), never to
replace it -- see scripts/train_ml_detector.py and
scripts/evaluate_hybrid.py for training and validation.

Architecture and training choices are deliberately different from the
approach in a comparable project (ICC) that we reviewed and found had
collapsed after epoch 1 on this same dataset (AISD), landing at 2.7%
precision:
  1. Loss here is BCE-with-pos-weight + soft Dice, not plain BCE -- plain
     BCE on a ~1%-positive-voxel class balance gives almost no gradient
     signal once the model learns to predict all-background, which is the
     textbook cause of an early collapse like that.
  2. The 3rd input channel is our own rule-based symmetry-difference map
     (detection.py's difference_map), not just raw mirrored CT -- gives
     the network the same prior signal the classical detector already
     uses successfully, rather than learning left-right symmetry from
     scratch on ~280 training patients.
This is still a small, fast model (SmallUNet below, ~16/32/64 channels) --
"lightweight" is a real design constraint here, not just a description.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """3-channel in (CT, mirrored CT, symmetry-diff), 1-channel logit out."""

    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = DoubleConv(64 + 32, 32)
        self.up1 = DoubleConv(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        u2 = nn.functional.interpolate(c3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, c2], dim=1))
        u1 = nn.functional.interpolate(u2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, c1], dim=1))
        return self.head(u1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * intersection + eps) / (denom + eps)).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + dice_loss(logits, target)


def build_input_channels(ct_slice: np.ndarray, mirrored_slice: np.ndarray, diff_slice: np.ndarray) -> np.ndarray:
    """Stack (CT, mirrored CT, symmetry-diff) into a (3, H, W) float32 array."""
    return np.stack([ct_slice, mirrored_slice, diff_slice], axis=0).astype(np.float32)


def predict_slice(model: nn.Module, ct_slice: np.ndarray, mirrored_slice: np.ndarray,
                   diff_slice: np.ndarray, device: torch.device, image_size: int = 256) -> np.ndarray:
    """Run the model on one 2D slice, return a probability map at the
    slice's original resolution."""
    model.eval()
    original_shape = ct_slice.shape
    x = build_input_channels(ct_slice, mirrored_slice, diff_slice)
    x_t = torch.from_numpy(x).unsqueeze(0)
    x_t = nn.functional.interpolate(x_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        logits = model(x_t.to(device))
        probs = torch.sigmoid(logits)
        probs = nn.functional.interpolate(probs, size=original_shape, mode="bilinear", align_corners=False)
    return probs.squeeze().cpu().numpy()
