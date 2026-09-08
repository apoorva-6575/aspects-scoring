"""CLI: run the full pipeline on one NCCT volume against the atlas.

Usage:
    python scripts/run_pipeline.py --patient path/to/patient.nii.gz \
        --atlas path/to/atlas_ncct.nii.gz --atlas-labels path/to/atlas_labels.nii.gz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import detect_ischemic_change
from src.registration import register_atlas_to_patient, warp_atlas_labels
from src.scoring import region_flags, mark_uncertain_near_boundaries, compute_aspects_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--atlas-labels", required=True)
    parser.add_argument("--percentile", type=float, default=90)
    parser.add_argument("--min-blob-voxels", type=int, default=15)
    parser.add_argument("--already-windowed", action="store_true",
                        help="Set this for AISD-derived volumes (see "
                             "scripts/convert_aisd_to_nifti.py) -- their pixels are "
                             "pre-windowed 0-255 display values, not raw HU.")
    args = parser.parse_args()

    print(f"[1/4] Preprocessing {args.patient}")
    pre = preprocess_volume(args.patient, already_windowed=args.already_windowed)

    print("[2/4] Detecting ischemic change (symmetry difference)")
    diff_map, change_mask = detect_ischemic_change(
        pre["windowed"], pre["brain_mask"],
        percentile=args.percentile, min_blob_voxels=args.min_blob_voxels,
    )
    print(f"  flagged voxels: {change_mask.sum()}")

    print("[3/4] Registering atlas -> patient")
    transform, metric = register_atlas_to_patient(args.atlas, args.patient)
    print(f"  registration metric (lower is usually better, calibrate this): {metric:.4f}")
    region_labels = warp_atlas_labels(args.atlas_labels, args.patient, transform)
    # NOTE: SimpleITK arrays come back as (z, y, x); nibabel volumes are (x, y, z).
    # Transpose to match before combining -- verify orientation against a known case.
    region_labels = region_labels.transpose(2, 1, 0)

    print("[4/4] Scoring")
    flags = region_flags(change_mask, region_labels)
    flags = mark_uncertain_near_boundaries(flags, region_labels, change_mask)
    score, flagged_count = compute_aspects_score(flags)

    print(f"\nASPECTS score: {score}/10  ({flagged_count} regions flagged)")
    for region_id, info in flags.items():
        marker = "FLAGGED" if info["flagged"] else "ok"
        note = f"  [{info['note']}]" if info["note"] else ""
        print(f"  {region_id:2d} {info['name']:<18} coverage={info['coverage']:.2f}  {marker}{note}")


if __name__ == "__main__":
    main()
