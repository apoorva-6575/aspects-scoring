"""Collect the registration-metric distribution (bg_metric, sc_metric)
across a sample of AISD patients, to pick a data-driven
REGISTRATION_CONFIDENCE_THRESHOLD for scoring.py.

There's no ground-truth "good vs bad registration" label in AISD, so this
can't be calibrated against a true accuracy target the way detection
thresholds were (scripts/tune_detection.py, which had AISD's lesion masks
as ground truth). Instead this picks a threshold from the metric's own
distribution: flag the worst quartile of registrations as low-confidence.
That's a reasonable default, not a validated accuracy guarantee -- if
real registration-quality ground truth ever becomes available (e.g. a few
manually-reviewed cases), recalibrate against that instead.

Usage:
    python scripts/calibrate_registration_confidence.py --n-patients 40
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.registration import register_aspects_atlas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--atlas-dir", default="data/atlas")
    parser.add_argument("--age-group", default="50_69")
    parser.add_argument("--n-patients", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir)
    bgl_image = atlas_dir / f"BGL_image_{args.age_group}.nii.gz"
    bgl_label = atlas_dir / f"BGL_label_{args.age_group}.nii.gz"
    sgl_image = atlas_dir / f"SGL_image_{args.age_group}.nii.gz"
    sgl_label = atlas_dir / f"SGL_label_{args.age_group}.nii.gz"

    patient_dirs = sorted(glob.glob(str(Path(args.nifti_dir) / "*")))
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(patient_dirs, size=min(args.n_patients, len(patient_dirs)), replace=False)

    bg_metrics, sc_metrics = [], []
    print(f"Running registration on {len(sample)} patients...")
    for i, pdir in enumerate(sample, 1):
        pdir = Path(pdir)
        image_path = pdir / "image.nii.gz"
        if not image_path.exists():
            continue
        pre = preprocess_volume(str(image_path), already_windowed=True)
        patient_spacing = (abs(pre["affine"][0, 0]), abs(pre["affine"][1, 1]))
        try:
            reg = register_aspects_atlas(
                pre["windowed"], pre["brain_mask"], patient_spacing,
                str(bgl_image), str(bgl_label), str(sgl_image), str(sgl_label),
            )
        except Exception as error:
            print(f"  {pdir.name}: FAILED ({error})")
            continue
        bg_metrics.append(reg["bg_metric"])
        sc_metrics.append(reg["sc_metric"])
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    bg_metrics = np.array(bg_metrics)
    sc_metrics = np.array(sc_metrics)
    all_metrics = np.concatenate([bg_metrics, sc_metrics])

    print(f"\nCollected {len(bg_metrics)} BG + {len(sc_metrics)} SC metrics "
          f"({len(all_metrics)} total).")
    print(f"BG metric: mean={bg_metrics.mean():.4f} std={bg_metrics.std():.4f} "
          f"min={bg_metrics.min():.4f} max={bg_metrics.max():.4f}")
    print(f"SC metric: mean={sc_metrics.mean():.4f} std={sc_metrics.std():.4f} "
          f"min={sc_metrics.min():.4f} max={sc_metrics.max():.4f}")

    for pct in [10, 25, 50, 75, 90]:
        print(f"  combined p{pct}: {np.percentile(all_metrics, pct):.4f}")

    # Lower (more negative) Mattes MI cost is better. Flag the worst quartile
    # (i.e. the least-negative/highest 25%) as low-confidence.
    threshold = float(np.percentile(all_metrics, 75))
    print(f"\nSuggested REGISTRATION_CONFIDENCE_THRESHOLD (worst-quartile cutoff): "
          f"{threshold:.4f}")
    print("A slice's metric ABOVE this (worse/less-negative) should be treated as low-confidence.")


if __name__ == "__main__":
    main()
