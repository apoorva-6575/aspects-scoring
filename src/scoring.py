"""Phase 4: combine Objective 1 (change mask) + Objective 2 (warped atlas
regions) into per-region flags and the final ASPECTS score.
"""

import numpy as np

# Confirmed against the ASPECTS-281 atlas (figshare.com/articles/figure/
# ASPECTS-281/26819290) and its source paper (PMC11480093): C=1, L=2, IC=3,
# I=4, M1-M6=5-10. BGL_label_*.nii.gz contains labels 1-7 (one 2D slice at
# basal-ganglia level); SGL_label_*.nii.gz contains labels 8-10 (a separate
# 2D slice at supraganglionic level) -- see registration.py notes on the 2D
# two-slice atlas structure. Left/right hemisphere is inferred from
# x-position, not a separate label id.
ASPECTS_REGIONS = {
    1: "Caudate",
    2: "Lentiform nucleus",
    3: "Internal capsule",
    4: "Insula",
    5: "M1",
    6: "M2",
    7: "M3",
    8: "M4",
    9: "M5",
    10: "M6",
}

# A region is flagged if this fraction of its voxels fall inside the
# detected change mask. Tune against AISD lesion masks (there's no
# ASPECTS-score ground truth in AISD, only lesion masks, so this can only
# be calibrated for the detection step, not the final score).
DEFAULT_FLAG_FRACTION = 0.15

# Below this registration metric quality, mark affected regions "uncertain"
# instead of confidently flagging them. Sign/scale depends on the metric
# used (Mattes MI cost is negative-is-better in SimpleITK) -- calibrate
# this threshold empirically on a handful of known-good vs. known-bad cases.
REGISTRATION_CONFIDENCE_THRESHOLD = None


def region_flags(change_mask, region_labels, flag_fraction=DEFAULT_FLAG_FRACTION):
    """For each ASPECTS region id, compute the fraction of the region
    covered by the change mask and whether it's flagged.

    Works on 2D slices or 3D volumes -- shape-agnostic. In this pipeline
    it's called once per ASPECTS slice (BG level, SC level), since each
    slice's warped region_labels only contains the region ids that atlas
    level covers (regions absent from a slice come back as "region not
    found in warp"); merge_slice_flags combines the two calls into one
    10-region result. `region_labels` must be the same shape/grid as
    `change_mask` (i.e. warped atlas output from registration.warp_2d_labels).
    """
    results = {}
    for region_id, name in ASPECTS_REGIONS.items():
        region_mask = region_labels == region_id
        region_size = region_mask.sum()
        if region_size == 0:
            results[region_id] = {
                "name": name, "coverage": 0.0, "flagged": False, "note": "region not found in warp"
            }
            continue
        coverage = float((region_mask & change_mask).sum()) / float(region_size)
        results[region_id] = {
            "name": name,
            "coverage": coverage,
            "flagged": coverage >= flag_fraction,
            "note": "",
        }
    return results


def mark_uncertain_near_boundaries(flags, region_labels, change_mask, boundary_dilation=2):
    """Flag regions as uncertain if the change mask sits right on a
    region boundary -- the exact failure mode the brief calls out: small
    registration error flipping which region a detection lands in.
    """
    from scipy import ndimage

    for region_id, info in flags.items():
        region_mask = region_labels == region_id
        if region_mask.sum() == 0:
            continue
        boundary = region_mask ^ ndimage.binary_erosion(region_mask, iterations=boundary_dilation)
        if (boundary & change_mask).any() and info["coverage"] < DEFAULT_FLAG_FRACTION * 1.5:
            info["note"] = "change detected near region boundary -- verify manually"
    return flags


def merge_slice_flags(bg_flags, sc_flags):
    """Combine region_flags() results from the BG-level and SC-level
    slices into one 10-region dict. Each region id is only real in one of
    the two slices (1-7 in BG, 8-10 in SC) -- prefer whichever result
    actually found the region, falling back to the other if neither did.
    """
    merged = {}
    for region_id in ASPECTS_REGIONS:
        bg_info = bg_flags.get(region_id)
        sc_info = sc_flags.get(region_id)
        for info in (bg_info, sc_info):
            if info is not None and info["note"] != "region not found in warp":
                merged[region_id] = info
                break
        else:
            merged[region_id] = bg_info or sc_info
    return merged


def compute_aspects_score(flags):
    """ASPECTS score = 10 - number of flagged regions."""
    flagged_count = sum(1 for f in flags.values() if f["flagged"])
    return max(0, 10 - flagged_count), flagged_count
