"""Convert AISD's per-patient PNG slice folders into NIfTI volumes.

AISD ships as `image/image/<patient_id>/000.png, 001.png, ...` (already
brain-windowed 8-bit grayscale, NOT raw HU) and matching
`mask/mask/<patient_id>/*.png` label maps with pixel values:
    0 = background
    1 = remote infarct
    2 = clear acute infarct
    3 = blurred acute infarct
    4 = invisible acute infarct  (visible on DWI/MRI only, NOT on CT)
    5 = infarct

Per the dataset authors, {1, 2, 3, 5} together form the "infarct lesion"
ground truth. Label 4 is explicitly *invisible on CT* -- it exists because
AISD's masks were drawn using DWI as reference, not because it's detectable
in the NCCT pixels. Since this pipeline's Objective 1 detector only looks
at NCCT, label 4 is excluded from the binary ground-truth mask used for
validation; scoring the detector against it would be an unfair target it
structurally cannot see.

No true voxel spacing is available from the PNGs (that lives in the
original DICOM headers, not extracted here) -- this uses an approximate
spacing which is fine for running the pipeline and eyeballing overlays,
but NOT for anything requiring real-world distances. Pull spacing from the
dicom-*.tar.gz files instead if that ever matters.

Usage:
    python scripts/convert_aisd_to_nifti.py \
        --aisd-dir data/aisd --out-dir data/aisd_nifti [--limit 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

VISIBLE_ON_CT_LABELS = {1, 2, 3, 5}
APPROX_IN_PLANE_SPACING_MM = 0.45
APPROX_SLICE_SPACING_MM = 5.0


def _load_slice_stack(patient_dir: Path) -> np.ndarray:
    slice_files = sorted(patient_dir.glob("*.png"), key=lambda p: p.stem)
    slices = [np.array(Image.open(f)) for f in slice_files]
    # PNG slices come in as (y, x) each; stack along z -> (y, x, z), then
    # move to (x, y, z) to match the rest of the pipeline's convention.
    volume = np.stack(slices, axis=-1)
    return np.transpose(volume, (1, 0, 2))


def _make_affine():
    return np.diag([
        APPROX_IN_PLANE_SPACING_MM,
        APPROX_IN_PLANE_SPACING_MM,
        APPROX_SLICE_SPACING_MM,
        1.0,
    ])


def convert_patient(image_dir: Path, mask_dir: Path, out_dir: Path, patient_id: str):
    image_vol = _load_slice_stack(image_dir).astype(np.uint8)
    affine = _make_affine()

    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image_vol, affine), out_dir / "image.nii.gz")

    if mask_dir.exists():
        mask_vol = _load_slice_stack(mask_dir).astype(np.uint8)
        if mask_vol.shape != image_vol.shape:
            print(f"  WARNING [{patient_id}]: mask shape {mask_vol.shape} != "
                  f"image shape {image_vol.shape}, skipping mask")
        else:
            nib.save(nib.Nifti1Image(mask_vol, affine), out_dir / "mask_multiclass.nii.gz")
            binary_mask = np.isin(mask_vol, list(VISIBLE_ON_CT_LABELS)).astype(np.uint8)
            nib.save(nib.Nifti1Image(binary_mask, affine), out_dir / "mask_binary.nii.gz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aisd-dir", default="data/aisd",
                        help="Folder containing image/image/<id> and mask/mask/<id>")
    parser.add_argument("--out-dir", default="data/aisd_nifti")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only convert the first N patients (useful for a quick test)")
    args = parser.parse_args()

    aisd_dir = Path(args.aisd_dir)
    out_dir = Path(args.out_dir)
    image_root = aisd_dir / "image" / "image"
    mask_root = aisd_dir / "mask" / "mask"

    if not image_root.exists():
        print(f"ERROR: {image_root} not found. Did you unzip image.zip under {aisd_dir}?")
        sys.exit(1)

    patient_dirs = sorted(p for p in image_root.iterdir() if p.is_dir())
    if args.limit:
        patient_dirs = patient_dirs[: args.limit]

    print(f"Converting {len(patient_dirs)} patients -> {out_dir}")
    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name
        convert_patient(patient_dir, mask_root / patient_id, out_dir / patient_id, patient_id)
        if i % 25 == 0 or i == len(patient_dirs):
            print(f"  {i}/{len(patient_dirs)} done")

    print("Done. Each patient folder has image.nii.gz, mask_multiclass.nii.gz, "
          "mask_binary.nii.gz (mask_binary excludes label 4 -- see module docstring).")


if __name__ == "__main__":
    main()
