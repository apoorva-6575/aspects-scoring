# ASPECTS Auto-Scoring

Rule-based / no-training pipeline for the IIC 3.0 problem statement:
detect early ischemic change on NCCT, localize it to the 10 ASPECTS
regions via atlas registration, and output an explainable 0-10 score.

No model training required — detection is symmetry-based (mirror across
midline, threshold the difference), and region localization is classical
image registration (SimpleITK), not a trained network.

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

Also grab the **NCCT ASPECTS atlas** (10 labeled regions) — see
[BravoSun/NCCT-atlas-for-ASPECTS-scoring](https://github.com/BravoSun/NCCT-atlas-for-ASPECTS-scoring)
or the [MIPLAB-NCCT atlas on Figshare](https://figshare.com/s/9a0ae1773fbf7f46347d).
You need three files to run anything:
- a patient NCCT volume (`.nii.gz`) — from AISD, see conversion step below
- the atlas NCCT volume (`.nii.gz`)
- the atlas region-label volume (`.nii.gz`, integer labels 1-10)

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
| `src/registration.py` | 3 (Objective 2) | rigid→affine→BSpline atlas registration, label warping |
| `src/scoring.py` | 4 (interface) | per-region flags, boundary-uncertainty marking, final score |
| `src/visualize.py` | 5 | slice overlay rendering |
| `app/streamlit_app.py` | 5 | demo UI |
| `scripts/run_pipeline.py` | — | CLI, run the whole thing on one volume without the UI |

## Run

CLI (fastest way to sanity-check the pipeline on one case):
```
python scripts/run_pipeline.py --patient data/patient1.nii.gz \
    --atlas data/atlas_ncct.nii.gz --atlas-labels data/atlas_labels.nii.gz
```

Demo UI:
```
streamlit run app/streamlit_app.py
```

## Known TODOs before this is demo-ready

- `src/scoring.py`: `ASPECTS_REGIONS` label-id mapping is a placeholder —
  fill in with the actual integer labels used by whichever atlas you
  download.
- `src/detection.py`: `mirror_across_x` assumes the volume is already
  roughly axis-aligned. If your dataset has tilted scans, rotate using
  `lr_axis` from `preprocessing.find_midline_axis` first.
- Threshold/percentile values (`detection.py`, `scoring.py`) are untuned —
  calibrate the detection mask against AISD's lesion masks (Dice score) to
  pick reasonable defaults (see build plan Phase 6). There's no ground
  truth for the final ASPECTS score itself, only for the lesion mask.
- Registration confidence threshold in `scoring.py` is unset — run a few
  known-good vs. known-bad registrations to calibrate it, then use it to
  gate the "uncertain" flag.
- SimpleITK arrays are `(z, y, x)`; nibabel volumes are `(x, y, z)` — both
  `run_pipeline.py` and `streamlit_app.py` transpose to reconcile this;
  double check against a known landmark before trusting it blindly.
