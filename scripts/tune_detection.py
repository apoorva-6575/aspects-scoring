"""Grid-search detection parameters (erode_iterations, percentile,
min_blob_voxels) against AISD's mask_binary ground truth, scoring by Dice.

Processes one patient at a time and accumulates running Dice sums per
combo -- does NOT cache all patients' diff maps in memory (an earlier
version did, and blew up to several GB across 60 patients x 5 erosion
levels, crashing with an allocation error partway through).

Usage:
    python scripts/tune_detection.py --nifti-dir data/aisd_nifti --n-patients 60
"""

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import mirror_across_x, difference_map, threshold_mask


def dice(pred, gt):
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * inter / denom if denom > 0 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--n-patients", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patient_dirs = sorted(glob.glob(str(Path(args.nifti_dir) / "*")))
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(patient_dirs, size=min(args.n_patients, len(patient_dirs)), replace=False)

    erode_options = [0, 2, 3, 4, 5]
    percentiles = [70, 75, 80, 85, 90]
    min_blobs = [10, 20, 40, 80]

    dice_sum = defaultdict(float)
    dice_count = defaultdict(int)
    n_used = 0

    print(f"Evaluating {len(sample)} patients x {len(erode_options)} erosion levels "
          f"x {len(percentiles)}x{len(min_blobs)} threshold combos...")
    for i, pdir in enumerate(sample, 1):
        pdir = Path(pdir)
        gt = nib.load(pdir / "mask_binary.nii.gz").get_fdata() > 0
        if gt.sum() == 0:
            continue
        pre = preprocess_volume(str(pdir / "image.nii.gz"), already_windowed=True)
        mirrored = mirror_across_x(pre["windowed"], brain_mask=pre["brain_mask"])

        for e in erode_options:
            eroded_mask = (ndimage.binary_erosion(pre["brain_mask"], iterations=e)
                           if e > 0 else pre["brain_mask"])
            diff = difference_map(pre["windowed"], mirrored, eroded_mask)
            for pct in percentiles:
                for blob in min_blobs:
                    pred = threshold_mask(diff, eroded_mask, pct, blob)
                    key = (e, pct, blob)
                    dice_sum[key] += dice(pred, gt)
                    dice_count[key] += 1
            del diff
        del pre, mirrored
        n_used += 1
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    results = {k: dice_sum[k] / dice_count[k] for k in dice_sum}
    ranked = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nTop 15 (erode_iterations, percentile, min_blob_voxels) by mean Dice "
          f"(over {n_used} patients):")
    for (e, pct, blob), score in ranked[:15]:
        print(f"  erode={e}  percentile={pct:>3}  min_blob_voxels={blob:>3}   mean Dice={score:.4f}")

    best_e, best_pct, best_blob = ranked[0][0]
    print(f"\nBest: erode_iterations={best_e}, percentile={best_pct}, "
          f"min_blob_voxels={best_blob}, mean Dice={ranked[0][1]:.4f}")


if __name__ == "__main__":
    main()
