"""CLI: run the full pipeline on one NCCT volume against the atlas.

Usage:
    python scripts/run_pipeline.py --patient data/aisd_nifti/<id>/image.nii.gz \
        --already-windowed --atlas-dir data/atlas --age-group 50_69
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import detect_ischemic_change
from src.registration import register_aspects_atlas
from src.scoring import region_flags, mark_uncertain_near_boundaries, merge_slice_flags, compute_aspects_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--atlas-dir", required=True,
                        help="Directory containing BGL_image_<age>.nii.gz, "
                             "BGL_label_<age>.nii.gz, SGL_image_<age>.nii.gz, "
                             "SGL_label_<age>.nii.gz (see README download step)")
    parser.add_argument("--age-group", default="50_69",
                        choices=["10_29", "30_49", "50_69", "70_89"])
    parser.add_argument("--percentile", type=float, default=70)
    parser.add_argument("--min-blob-voxels", type=int, default=80)
    parser.add_argument("--erode-iterations", type=int, default=3)
    parser.add_argument("--already-windowed", action="store_true",
                        help="Set this for AISD-derived volumes (see "
                             "scripts/convert_aisd_to_nifti.py) -- their pixels are "
                             "pre-windowed 0-255 display values, not raw HU.")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir)
    bgl_image = atlas_dir / f"BGL_image_{args.age_group}.nii.gz"
    bgl_label = atlas_dir / f"BGL_label_{args.age_group}.nii.gz"
    sgl_image = atlas_dir / f"SGL_image_{args.age_group}.nii.gz"
    sgl_label = atlas_dir / f"SGL_label_{args.age_group}.nii.gz"

    print(f"[1/4] Preprocessing {args.patient}")
    pre = preprocess_volume(args.patient, already_windowed=args.already_windowed)

    print("[2/4] Detecting ischemic change (symmetry difference)")
    diff_map, change_mask = detect_ischemic_change(
        pre["windowed"], pre["brain_mask"],
        percentile=args.percentile, min_blob_voxels=args.min_blob_voxels,
        erode_iterations=args.erode_iterations,
    )
    print(f"  flagged voxels (whole volume): {change_mask.sum()}")

    print(f"[3/4] Registering ASPECTS atlas (age group {args.age_group}) -> patient")
    reg = register_aspects_atlas(pre["windowed"], pre["brain_mask"],
                                  str(bgl_image), str(bgl_label), str(sgl_image), str(sgl_label))
    print(f"  BG-level slice: z={reg['bg_slice_idx']}  metric={reg['bg_metric']:.4f}")
    print(f"  SC-level slice: z={reg['sc_slice_idx']}  metric={reg['sc_metric']:.4f}")

    print("[4/4] Scoring")
    bg_change_2d = change_mask[:, :, reg["bg_slice_idx"]]
    sc_change_2d = change_mask[:, :, reg["sc_slice_idx"]]

    bg_flags = region_flags(bg_change_2d, reg["bg_region_labels"])
    bg_flags = mark_uncertain_near_boundaries(bg_flags, reg["bg_region_labels"], bg_change_2d)
    sc_flags = region_flags(sc_change_2d, reg["sc_region_labels"])
    sc_flags = mark_uncertain_near_boundaries(sc_flags, reg["sc_region_labels"], sc_change_2d)

    flags = merge_slice_flags(bg_flags, sc_flags)
    score, flagged_count = compute_aspects_score(flags)

    print(f"\nASPECTS score: {score}/10  ({flagged_count} regions flagged)")
    for region_id, info in flags.items():
        marker = "FLAGGED" if info["flagged"] else "ok"
        note = f"  [{info['note']}]" if info["note"] else ""
        print(f"  {region_id:2d} {info['name']:<18} coverage={info['coverage']:.2f}  {marker}{note}")


if __name__ == "__main__":
    main()
