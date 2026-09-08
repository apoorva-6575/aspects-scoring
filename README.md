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

Download and register for **APIS** (primary dataset), and grab the
**NCCT ASPECTS atlas** (10 labeled regions) from the resource links on the
problem statement page. You need three files to run anything:
- a patient NCCT volume (`.nii.gz`)
- the atlas NCCT volume (`.nii.gz`)
- the atlas region-label volume (`.nii.gz`, integer labels 1-10)

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
  calibrate against APIS ground truth (see build plan Phase 6).
- Registration confidence threshold in `scoring.py` is unset — run a few
  known-good vs. known-bad registrations to calibrate it, then use it to
  gate the "uncertain" flag.
- SimpleITK arrays are `(z, y, x)`; nibabel volumes are `(x, y, z)` — both
  `run_pipeline.py` and `streamlit_app.py` transpose to reconcile this;
  double check against a known landmark before trusting it blindly.
