"""Create a reproducible train/validation/test split over AISD patients,
for training the optional ML detector (src/ml_detector.py) and evaluating
it on genuinely held-out data.

We didn't have a split before this -- all prior calibration (tune_detection.py,
calibrate_registration_confidence.py) sampled randomly from the full 398
patients with no train/test separation, which is fine for picking
classical-method *thresholds* (a handful of numbers, not learned weights)
but not appropriate for a trained model, where evaluating on
training-adjacent data would overstate real performance.

Usage:
    python scripts/split_dataset.py --nifti-dir data/aisd_nifti --output data/split.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", default="data/aisd_nifti")
    parser.add_argument("--output", default="data/split.json")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    patient_ids = sorted(Path(p).name for p in glob.glob(str(Path(args.nifti_dir) / "*")))
    rng = np.random.default_rng(args.seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    split = {
        "train": sorted(shuffled[:n_train]),
        "validation": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
        "seed": args.seed,
    }

    Path(args.output).write_text(json.dumps(split, indent=2))
    print(f"train={len(split['train'])} validation={len(split['validation'])} test={len(split['test'])}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
