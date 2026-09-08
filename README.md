# ASPECTS Auto-Scoring

Rule-based / no-training pipeline for the IIC 3.0 problem statement:
detect early ischemic change on NCCT, localize it to the 10 ASPECTS
regions via atlas registration, and output an explainable 0-10 score.

No model training required — detection is symmetry-based (mirror across
midline, threshold the difference), and region localization is classical
image registration (SimpleITK), not a trained network.

Region localization uses a **2D two-slice atlas**, not a 3D volume — this
mirrors real clinical ASPECTS practice, where radiologists score from
exactly two representative axial slices (basal ganglia level, BGL; and
supraganglionic level, SGL), not the whole brain. See `src/registration.py`
for the slice-selection + per-slice registration logic.

## Setup

```
pip install -r requirements.txt
```

Dataset: **AISD** (397 NCCT scans with DWI-confirmed lesion masks, no
registration required — https://github.com/GriffinLiang/AISD). APIS would
have been the primary choice (paired CT-MRI, has cleaner annotations) but
its registration/access isn't public, so AISD is what this pipeline is
built against. Note the tradeoff: AISD has lesion masks but no ASPECTS
region/score labels, so you can validate the Objective 1 detection mask
(Dice against AISD masks) but the final 0-10 score has no ground truth to
check against — treat that part as qualitative validation only.

Also grab the **ASPECTS-281 atlas** (age-specific, 10 labeled regions,
figshare.com/articles/figure/ASPECTS-281/26819290 — code/methodology at
[BravoSun/NCCT-atlas-for-ASPECTS-scoring](https://github.com/BravoSun/NCCT-atlas-for-ASPECTS-scoring)).
It ships as 4 files per age group (`10_29`, `30_49`, `50_69`, `70_89`),
not one bundle — fetch via the Figshare API (the article page itself
403s to scrapers):

```
curl -s "https://api.figshare.com/v2/articles/26819290" -H "User-Agent: Mozilla/5.0" \
  | python -c "import json,sys; [print(f['name'], f['download_url']) for f in json.load(sys.stdin)['files']]"
```

then `curl -sL -A "Mozilla/5.0" -o <name> <download_url>` for the age
group you want (this repo's example commands below use `50_69`) into
`data/atlas/`. You need 4 files to run anything:
- a patient NCCT volume (`.nii.gz`) — from AISD, see conversion step below
- `BGL_image_<age>.nii.gz` / `BGL_label_<age>.nii.gz` — basal ganglia level
- `SGL_image_<age>.nii.gz` / `SGL_label_<age>.nii.gz` — supraganglionic level

Label encoding (confirmed against the source paper, PMC11480093): in the
BGL file, `Caudate=1, Lentiform=2, Internal capsule=3, Insula=4, M1=5,
M2=6, M3=7`; in the SGL file, `M4=8, M5=9, M6=10`. This is also hardcoded
in `src/scoring.py::ASPECTS_REGIONS`.

### Converting AISD to NIfTI

AISD ships as per-patient folders of 8-bit PNG slices
(`image/image/<id>/000.png, 001.png, ...`), already brain-windowed for
display — **not** raw Hounsfield units, and not NIfTI. Convert before
running the pipeline:

```
python scripts/convert_aisd_to_nifti.py --aisd-dir data/aisd --out-dir data/aisd_nifti
```

This writes `image.nii.gz`, `mask_multiclass.nii.gz`, and `mask_binary.nii.gz`
per patient. The mask has 5 label values (see the script's docstring);
`mask_binary` combines the CT-visible ones ({1,2,3,5}) and drops label 4
(infarct visible on DWI but *invisible on CT* — an unfair target for a
detector that only looks at NCCT). Because these images are pre-windowed,
not raw HU, load them with `preprocess_volume(path, already_windowed=True)`
(`src/preprocessing.py`) — passing `already_windowed=False` (the default,
meant for raw-HU NIfTI like the atlas) on AISD-derived volumes will apply
the wrong windowing and skull-strip thresholds.

No real voxel spacing is available from the PNGs (that's in AISD's DICOM
files, not extracted here) — the converter assumes an approximate spacing.
Fine for running the pipeline and demoing overlays; don't trust it for
anything requiring true physical distances.

## Structure

| File | Phase | What it does |
|---|---|---|
| `src/preprocessing.py` | 1 | load, HU windowing, skull strip, midline axis |
| `src/detection.py` | 2 (Objective 1) | mirror across midline, diff map, threshold → change mask |
| `src/registration.py` | 3 (Objective 2) | pick BG/SC slices, rigid→affine 2D atlas registration, label warping |
| `src/scoring.py` | 4 (interface) | per-region flags, boundary-uncertainty marking, final score |
| `src/visualize.py` | 5 | slice overlay rendering |
| `app/streamlit_app.py` | 5 | demo UI |
| `scripts/run_pipeline.py` | — | CLI, run the whole thing on one volume without the UI |

## Run

CLI (fastest way to sanity-check the pipeline on one case):
```
python scripts/run_pipeline.py --patient data/aisd_nifti/<id>/image.nii.gz \
    --already-windowed --atlas-dir data/atlas --age-group 50_69
```

Demo UI:
```
streamlit run app/streamlit_app.py
```

## Known TODOs before this is demo-ready

- `src/registration.py::select_aspects_slices`: BG/SC slice indices are
  picked by a crude fixed-fraction-of-brain-height heuristic
  (`bg_fraction=0.40, sc_fraction=0.58`), not real anatomy detection.
  Verified end-to-end on one AISD case (picked z=7 and z=9 of 17 slices,
  registered regions landed centrally/plausibly in the brain) but these
  fractions are untuned — check against a handful of cases with known
  slice anatomy and adjust.
- `src/detection.py`: `mirror_across_x` assumes the volume is already
  roughly axis-aligned. If your dataset has tilted scans, rotate using
  `lr_axis` from `preprocessing.find_midline_axis` first.
- **Detection has been calibrated** (`scripts/tune_detection.py`, grid
  search over 60 AISD patients, results in `src/detection.py`'s
  docstring): `erode_iterations=3, percentile=70, min_blob_voxels=80`,
  mean Dice **0.073** against AISD's `mask_binary.nii.gz`. Two real bugs
  were found and fixed along the way: (1) `mirror_across_x` was mirroring
  around the image's geometric center instead of the brain's actual
  centroid, which differ by up to ~70px on AISD volumes — now auto-derived
  from `brain_mask`; (2) `threshold_mask` had an O(voxels × blobs) loop
  that made it ~7s/call, now vectorized to ~0.1s. Be honest about what
  0.073 Dice means: this is a genuinely hard detection problem on NCCT
  (the problem statement itself says the signal can be "only a few
  Hounsfield units"), and this is the best result found with a purely
  classical symmetry-difference method, not a solved detector. If there's
  time left, the highest-leverage next steps are: real anatomical
  landmark-based skull stripping (HD-BET) instead of the crude HU
  threshold, and/or restricting the difference map to a plausible
  parenchyma intensity band to exclude CSF/ventricle noise. There's still
  no ground truth for the final ASPECTS score itself, only for the lesion
  mask — the score has to be validated qualitatively.
- Re-run `scripts/tune_detection.py --n-patients <more>` if there's time
  to calibrate on a larger sample; 60 was chosen for speed, not because
  it's definitively enough.
- Registration confidence threshold in `scoring.py` is unset — run a few
  known-good vs. known-bad registrations to calibrate it, then use it to
  gate the "uncertain" flag.
