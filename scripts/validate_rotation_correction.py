"""Validate the tilt-correction fix (preprocessing.find_symmetry_rotation_angle
+ detection.mirror_across_x's rotation_deg) against a synthetic test case.

Real AISD volumes are apparently not tilted enough to exercise this path
naturally (see the "before/after" numbers this script prints on an
unmodified patient -- angle found is usually near 0). So this creates a
synthetic tilted version of a real patient by rotating both the image and
its ground-truth mask by a known angle, then checks:
  1. find_symmetry_rotation_angle recovers roughly that angle back
  2. detection Dice against the (correspondingly rotated) ground truth is
     better WITH the correction than without it

Usage:
    python scripts/validate_rotation_correction.py --patient 0019983 --tilt-deg 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume, find_symmetry_rotation_angle
from src.detection import detect_ischemic_change


def dice(pred, gt):
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * inter / denom if denom > 0 else 1.0


def rotate_volume(volume, angle_deg, order):
    return ndimage.rotate(volume, angle=angle_deg, axes=(0, 1), reshape=False,
                          order=order, mode="nearest" if order > 0 else "constant")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--patient", default="0019983")
    parser.add_argument("--tilt-deg", type=float, default=12.0)
    args = parser.parse_args()

    pdir = Path(args.nifti_dir) / args.patient
    print(f"[baseline] checking tilt already found on the unmodified patient...")
    pre_baseline = preprocess_volume(str(pdir / "image.nii.gz"), already_windowed=True)
    print(f"  detected angle: {pre_baseline['rotation_deg']:.1f} deg "
          f"(symmetry score {pre_baseline['rotation_symmetry_score']:.4f})")

    print(f"\n[synthetic] rotating patient {args.patient} by {args.tilt_deg} deg "
          f"to create a tilted test case...")
    raw_image = nib.load(str(pdir / "image.nii.gz")).get_fdata(dtype=np.float32)
    raw_mask = nib.load(str(pdir / "mask_binary.nii.gz")).get_fdata(dtype=np.float32)

    tilted_image = rotate_volume(raw_image, args.tilt_deg, order=1)
    tilted_gt = rotate_volume(raw_mask, args.tilt_deg, order=0) > 0.5

    # Re-run preprocessing on the tilted image (skull strip etc. must also
    # see the tilted version).
    from src.preprocessing import skull_strip_threshold
    tilted_windowed = np.clip(tilted_image, 0, 255) / 255.0
    tilted_brain_mask = skull_strip_threshold(tilted_image, 20, 230)
    found_angle, found_score = find_symmetry_rotation_angle(tilted_windowed, tilted_brain_mask)
    # The patient's own scan may already have a nonzero natural angle, so the
    # expected recovered angle is baseline + injected tilt, not the injected
    # tilt alone.
    expected = pre_baseline["rotation_deg"] + args.tilt_deg
    print(f"  find_symmetry_rotation_angle recovered: {found_angle:.1f} deg "
          f"(expected ~{expected:.1f} = baseline {pre_baseline['rotation_deg']:.1f} "
          f"+ injected {args.tilt_deg:.1f}, symmetry score {found_score:.4f})")

    print("\n[compare] detection Dice with vs. without rotation correction...")
    _, mask_uncorrected = detect_ischemic_change(tilted_windowed, tilted_brain_mask, rotation_deg=0.0)
    dice_uncorrected = dice(mask_uncorrected, tilted_gt)
    print(f"  rotation_deg=0.0 (no correction):        Dice={dice_uncorrected:.4f}")

    _, mask_corrected = detect_ischemic_change(tilted_windowed, tilted_brain_mask, rotation_deg=found_angle)
    dice_corrected = dice(mask_corrected, tilted_gt)
    print(f"  rotation_deg={found_angle:.1f} (corrected):        Dice={dice_corrected:.4f}")

    print(f"\nImprovement: {dice_corrected - dice_uncorrected:+.4f} Dice")


if __name__ == "__main__":
    main()
