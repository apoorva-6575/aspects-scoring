"""Compute Dice, Precision, Recall, and voxel-level Accuracy for the
current calibrated detection defaults, with rotation correction enabled.

Restricts "accuracy" to voxels inside the brain mask (not the whole image
canvas) -- even so, background (non-lesion) voxels vastly outnumber lesion
voxels, so accuracy will look deceptively high regardless of real
detection quality. That's a standard pitfall for imbalanced medical
segmentation; Dice/Precision/Recall are the metrics that actually reflect
performance here, and are reported alongside accuracy specifically so it
isn't read in isolation.

Usage:
    python scripts/compute_metrics.py --n-patients 60
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import nibabel as nib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import detect_ischemic_change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--n-patients", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patient_dirs = sorted(glob.glob(str(Path(args.nifti_dir) / "*")))
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(patient_dirs, size=min(args.n_patients, len(patient_dirs)), replace=False)

    total_tp = total_fp = total_fn = total_tn = 0
    per_case_dice = []
    n_used = 0

    print(f"Evaluating {len(sample)} patients (current calibrated defaults, rotation correction ON)...")
    for i, pdir in enumerate(sample, 1):
        pdir = Path(pdir)
        gt = nib.load(pdir / "mask_binary.nii.gz").get_fdata() > 0
        if gt.sum() == 0:
            continue

        pre = preprocess_volume(str(pdir / "image.nii.gz"), already_windowed=True)
        _, pred = detect_ischemic_change(
            pre["windowed"], pre["brain_mask"], rotation_deg=pre["rotation_deg"],
        )

        brain = pre["brain_mask"]
        pred_b, gt_b = pred[brain], gt[brain]

        tp = np.sum(pred_b & gt_b)
        fp = np.sum(pred_b & ~gt_b)
        fn = np.sum(~pred_b & gt_b)
        tn = np.sum(~pred_b & ~gt_b)

        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        denom = 2 * tp + fp + fn
        per_case_dice.append(2 * tp / denom if denom > 0 else 1.0)
        n_used += 1
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
    dice_pooled = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 1.0
    dice_mean = float(np.mean(per_case_dice))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n=== Results over {n_used} patients (voxels inside brain mask only) ===")
    print(f"Precision:        {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:           {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1:               {f1:.4f}")
    print(f"Dice (pooled):    {dice_pooled:.4f}")
    print(f"Dice (per-case mean): {dice_mean:.4f}")
    print(f"Accuracy:         {accuracy:.4f}  ({accuracy*100:.2f}%)  <- inflated by class imbalance, see docstring")
    print(f"\nRaw counts: TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
    print(f"Lesion voxels (positive class) are {100*total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0:.4f}% "
          f"recalled but only {100*(total_tp+total_fn)/(total_tp+total_fp+total_fn+total_tn):.4f}% "
          f"of all brain voxels -- that imbalance is why accuracy looks high regardless of Dice.")


if __name__ == "__main__":
    main()
