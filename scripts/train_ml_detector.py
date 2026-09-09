"""Train the optional ML detector (src/ml_detector.py) on AISD's train
split, validating on the held-out validation split each epoch.

Rotation correction (preprocessing.find_symmetry_rotation_angle) is
deliberately skipped during data loading here -- it costs ~9-15s per
patient (a 41-candidate angle search) and isn't needed to train useful
features, only to maximize final inference quality. Training data uses
rotation_deg=0 mirroring; the trained model is still evaluated (and used
at inference) against properly tilt-corrected slices later.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_ml_detector.py \
        --split data/split.json --epochs 6 --max-train-patients 100
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import skull_strip_threshold
from src.detection import mirror_across_x, difference_map
from src.ml_detector import SmallUNet, combined_loss, build_input_channels

IMAGE_SIZE = 256
VISIBLE_ON_CT_LABELS = {1, 2, 3, 5}


def load_patient_arrays(nifti_dir: Path, patient_id: str):
    image_path = nifti_dir / patient_id / "image.nii.gz"
    mask_path = nifti_dir / patient_id / "mask_multiclass.nii.gz"
    raw = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    windowed = np.clip(raw, 0, 255) / 255.0
    brain_mask = skull_strip_threshold(raw, 20, 230)
    mirrored = mirror_across_x(windowed, brain_mask=brain_mask, rotation_deg=0.0)
    diff = difference_map(windowed, mirrored, brain_mask)

    target = None
    if mask_path.exists():
        mask_raw = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
        target = np.isin(mask_raw, list(VISIBLE_ON_CT_LABELS)).astype(np.float32)
    return windowed, mirrored, diff, target


class AISDSliceDataset(Dataset):
    def __init__(self, nifti_dir: Path, patient_ids: list[str], cache_limit: int = 3):
        self.nifti_dir = nifti_dir
        self.slices: list[tuple[str, int]] = []
        for pid in patient_ids:
            mask_path = nifti_dir / pid / "mask_multiclass.nii.gz"
            if not mask_path.exists():
                continue
            n_slices = nib.load(str(mask_path)).shape[2]
            for z in range(n_slices):
                self.slices.append((pid, z))
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_limit = cache_limit

    def __len__(self):
        return len(self.slices)

    def _get_patient(self, pid: str):
        if pid not in self._cache:
            self._cache[pid] = load_patient_arrays(self.nifti_dir, pid)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(pid)
        return self._cache[pid]

    def __getitem__(self, idx):
        pid, z = self.slices[idx]
        windowed, mirrored, diff, target = self._get_patient(pid)
        x = build_input_channels(windowed[:, :, z], mirrored[:, :, z], diff[:, :, z])
        y = target[:, :, z][None, ...]
        x_t = torch.from_numpy(x).unsqueeze(0)
        y_t = torch.from_numpy(y).unsqueeze(0)
        x_t = torch.nn.functional.interpolate(x_t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        y_t = torch.nn.functional.interpolate(y_t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
        return x_t.squeeze(0), y_t.squeeze(0)


def dice_from_logits(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + 1.0) / (denom + 1.0)).mean()


def run_epoch(model, loader, optimizer, device, pos_weight, training: bool):
    model.train(training)
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = combined_loss(logits, y, pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss)
        total_dice += float(dice_from_logits(logits.detach(), y))
        n_batches += 1
    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--split", default="data/split.json")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-patients", type=int, default=None)
    parser.add_argument("--max-val-patients", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    split = json.loads(Path(args.split).read_text())
    train_ids = split["train"][: args.max_train_patients]
    val_ids = split["validation"][: args.max_val_patients]
    print(f"train patients: {len(train_ids)}, validation patients: {len(val_ids)}")

    train_ds = AISDSliceDataset(Path(args.nifti_dir), train_ids)
    val_ds = AISDSliceDataset(Path(args.nifti_dir), val_ids)
    print(f"train slices: {len(train_ds)}, validation slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class-weighted loss: lesion voxels are a tiny fraction of the brain,
    # so an unweighted loss lets the model minimize loss by predicting
    # all-background -- the standard cause of the "collapses immediately"
    # failure mode. Estimate the imbalance from a handful of training
    # slices rather than scanning the whole set (expensive, and a rough
    # estimate is enough to pick a reasonable weight).
    sample_pos, sample_total = 0.0, 0.0
    for i in range(min(30, len(train_ds))):
        _, y = train_ds[i]
        sample_pos += float(y.sum())
        sample_total += float(y.numel())
    pos_frac = max(sample_pos / sample_total, 1e-4)
    pos_weight = torch.tensor([min((1 - pos_frac) / pos_frac, 100.0)])
    print(f"estimated positive-voxel fraction: {pos_frac:.5f}, pos_weight: {pos_weight.item():.2f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, device, pos_weight, training=True)
        val_loss, val_dice = run_epoch(model, val_loader, optimizer, device, pos_weight, training=False)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
               "val_loss": val_loss, "val_dice": val_dice, "seconds": elapsed}
        history.append(row)
        print(json.dumps(row))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), output_dir / "best_ml_detector.pt")
            print(f"  -> new best (val_dice={val_dice:.4f}), checkpoint saved")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
