"""Phase 5: demo UI. Upload/select an NCCT volume, browse slices, see the
detected-change overlay, region boundaries, and the ASPECTS score.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_volume
from src.detection import detect_ischemic_change
from src.registration import register_atlas_to_patient, warp_atlas_labels
from src.scoring import region_flags, mark_uncertain_near_boundaries, compute_aspects_score
from src.visualize import render_slice

st.set_page_config(page_title="ASPECTS Auto-Scoring", layout="wide")
st.title("Automated ASPECTS Scoring")

with st.sidebar:
    st.header("Inputs")
    patient_path = st.text_input("Patient NCCT path (.nii.gz)")
    atlas_path = st.text_input("Atlas NCCT path (.nii.gz)")
    atlas_labels_path = st.text_input("Atlas region-labels path (.nii.gz)")
    percentile = st.slider("Detection sensitivity (percentile)", 80, 99, 90)
    min_blob = st.slider("Min blob size (voxels)", 1, 100, 15)
    run = st.button("Run pipeline")

if "result" not in st.session_state:
    st.session_state["result"] = None

if run and patient_path and atlas_path and atlas_labels_path:
    with st.spinner("Preprocessing..."):
        pre = preprocess_volume(patient_path)
    with st.spinner("Detecting ischemic change..."):
        diff_map, change_mask = detect_ischemic_change(
            pre["windowed"], pre["brain_mask"], percentile=percentile, min_blob_voxels=min_blob
        )
    with st.spinner("Registering atlas (this is the slow step)..."):
        transform, metric = register_atlas_to_patient(atlas_path, patient_path)
        region_labels = warp_atlas_labels(atlas_labels_path, patient_path, transform)
        region_labels = region_labels.transpose(2, 1, 0)
    with st.spinner("Scoring..."):
        flags = region_flags(change_mask, region_labels)
        flags = mark_uncertain_near_boundaries(flags, region_labels, change_mask)
        score, flagged_count = compute_aspects_score(flags)

    st.session_state["result"] = {
        "windowed": pre["windowed"],
        "change_mask": change_mask,
        "region_labels": region_labels,
        "flags": flags,
        "score": score,
        "flagged_count": flagged_count,
        "metric": metric,
    }

result = st.session_state["result"]

if result is None:
    st.info("Fill in the paths in the sidebar and click Run pipeline.")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        n_slices = result["windowed"].shape[2]
        slice_idx = st.slider("Slice", 0, n_slices - 1, n_slices // 2)
        fig = render_slice(
            result["windowed"][:, :, slice_idx],
            result["change_mask"][:, :, slice_idx],
            result["region_labels"][:, :, slice_idx],
            title=f"Slice {slice_idx}",
        )
        st.pyplot(fig)

    with col2:
        st.metric("ASPECTS score", f"{result['score']}/10")
        st.caption(f"{result['flagged_count']} region(s) flagged")
        st.caption(f"Registration metric: {result['metric']:.4f} (calibrate against known cases)")
        st.subheader("Per-region breakdown")
        for region_id, info in result["flags"].items():
            label = f"{region_id}. {info['name']}"
            if info["flagged"]:
                st.error(f"{label} — coverage {info['coverage']:.0%}")
            else:
                st.success(f"{label} — coverage {info['coverage']:.0%}")
            if info["note"]:
                st.caption(f"⚠ {info['note']}")
