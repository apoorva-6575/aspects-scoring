"""Phase 5: demo UI. Pick an NCCT volume, see the two ASPECTS-relevant
slices (basal ganglia level, supraganglionic level) with the detected
change overlay and region boundaries, plus the final score.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import detect_ischemic_change
from src.registration import register_aspects_atlas
from src.scoring import (
    region_flags, mark_uncertain_near_boundaries, mark_low_registration_confidence,
    merge_slice_flags, compute_aspects_score,
)
from src.visualize import render_slice

st.set_page_config(page_title="ASPECTS Auto-Scoring", layout="wide")
st.title("Automated ASPECTS Scoring")

with st.sidebar:
    st.header("Inputs")
    patient_path = st.text_input("Patient NCCT path (.nii.gz)")
    atlas_dir = st.text_input("Atlas directory", value="data/atlas")
    age_group = st.selectbox("Atlas age group", ["10_29", "30_49", "50_69", "70_89"], index=2)
    percentile = st.slider("Detection sensitivity (percentile)", 50, 99, 70)
    min_blob = st.slider("Min blob size (voxels)", 1, 150, 80)
    erode_iterations = st.slider(
        "Skull-strip boundary erosion (voxels)", 0, 8, 3,
        help="Shrinks the brain mask before detection to exclude the skull-strip "
             "boundary rim, which otherwise dominates the threshold with edge noise.",
    )
    already_windowed = st.checkbox(
        "Pre-windowed source (e.g. AISD-derived volume)",
        help="Check this for volumes from scripts/convert_aisd_to_nifti.py -- "
             "their pixels are 0-255 display values, not raw HU.",
    )
    run = st.button("Run pipeline")

if "result" not in st.session_state:
    st.session_state["result"] = None

if run and patient_path and atlas_dir:
    atlas_dir_path = Path(atlas_dir)
    bgl_image = atlas_dir_path / f"BGL_image_{age_group}.nii.gz"
    bgl_label = atlas_dir_path / f"BGL_label_{age_group}.nii.gz"
    sgl_image = atlas_dir_path / f"SGL_image_{age_group}.nii.gz"
    sgl_label = atlas_dir_path / f"SGL_label_{age_group}.nii.gz"

    with st.spinner("Preprocessing..."):
        pre = preprocess_volume(patient_path, already_windowed=already_windowed)
    with st.spinner("Detecting ischemic change..."):
        diff_map, change_mask = detect_ischemic_change(
            pre["windowed"], pre["brain_mask"], rotation_deg=pre["rotation_deg"],
            percentile=percentile, min_blob_voxels=min_blob, erode_iterations=erode_iterations,
        )
    with st.spinner("Registering ASPECTS atlas..."):
        patient_spacing = (abs(pre["affine"][0, 0]), abs(pre["affine"][1, 1]))
        reg = register_aspects_atlas(pre["windowed"], pre["brain_mask"], patient_spacing,
                                      str(bgl_image), str(bgl_label), str(sgl_image), str(sgl_label))
    with st.spinner("Scoring..."):
        bg_change_2d = change_mask[:, :, reg["bg_slice_idx"]]
        sc_change_2d = change_mask[:, :, reg["sc_slice_idx"]]

        bg_flags = region_flags(bg_change_2d, reg["bg_region_labels"])
        bg_flags = mark_uncertain_near_boundaries(bg_flags, reg["bg_region_labels"], bg_change_2d)
        bg_flags = mark_low_registration_confidence(bg_flags, reg["bg_metric"])
        sc_flags = region_flags(sc_change_2d, reg["sc_region_labels"])
        sc_flags = mark_uncertain_near_boundaries(sc_flags, reg["sc_region_labels"], sc_change_2d)
        sc_flags = mark_low_registration_confidence(sc_flags, reg["sc_metric"])

        flags = merge_slice_flags(bg_flags, sc_flags)
        score, flagged_count = compute_aspects_score(flags)

    st.session_state["result"] = {
        "windowed": pre["windowed"],
        "bg_slice_idx": reg["bg_slice_idx"],
        "sc_slice_idx": reg["sc_slice_idx"],
        "bg_change_2d": bg_change_2d,
        "sc_change_2d": sc_change_2d,
        "bg_region_labels": reg["bg_region_labels"],
        "sc_region_labels": reg["sc_region_labels"],
        "bg_metric": reg["bg_metric"],
        "sc_metric": reg["sc_metric"],
        "rotation_deg": pre["rotation_deg"],
        "flags": flags,
        "score": score,
        "flagged_count": flagged_count,
    }

result = st.session_state["result"]

if result is None:
    st.info("Fill in the paths in the sidebar and click Run pipeline.")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Basal ganglia level")
        fig_bg = render_slice(
            result["windowed"][:, :, result["bg_slice_idx"]],
            result["bg_change_2d"],
            result["bg_region_labels"],
            title=f"BG level (z={result['bg_slice_idx']}, metric={result['bg_metric']:.3f})",
        )
        st.pyplot(fig_bg)

        st.subheader("Supraganglionic level")
        fig_sc = render_slice(
            result["windowed"][:, :, result["sc_slice_idx"]],
            result["sc_change_2d"],
            result["sc_region_labels"],
            title=f"SC level (z={result['sc_slice_idx']}, metric={result['sc_metric']:.3f})",
        )
        st.pyplot(fig_sc)

    with col2:
        st.metric("ASPECTS score", f"{result['score']}/10")
        st.caption(f"{result['flagged_count']} region(s) flagged")
        st.caption(f"Detected scan tilt: {result['rotation_deg']:.1f}° "
                   "(corrected for automatically before mirroring)")
        st.caption("A slice with the worst-quartile registration metric (seen across 40 "
                   "calibration patients) is marked low-confidence below — a distribution-based "
                   "default, not validated accuracy.")
        st.subheader("Per-region breakdown")
        for region_id, info in result["flags"].items():
            label = f"{region_id}. {info['name']}"
            if info["flagged"]:
                st.error(f"{label} — coverage {info['coverage']:.0%}")
            else:
                st.success(f"{label} — coverage {info['coverage']:.0%}")
            if info["note"]:
                st.caption(f"⚠ {info['note']}")
