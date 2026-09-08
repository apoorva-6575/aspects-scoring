"""Phase 4: combine Objective 1 (change mask) + Objective 2 (warped atlas
regions) into per-region flags and the final ASPECTS score.
"""

import numpy as np

# TODO: fill in with the actual label values from the atlas you download --
# these are placeholders. Left/right pairs share the same label id in most
# NCCT ASPECTS atlases; hemisphere is inferred from x-position instead.
ASPECTS_REGIONS = {
    1: "Caudate",
    2: "Lentiform nucleus",
    3: "Insula",
    4: "Internal capsule",
    5: "M1",
    6: "M2",
    7: "M3",
    8: "M4",
    9: "M5",
    10: "M6",
}

# A region is flagged if this fraction of its voxels fall inside the
# detected change mask. Tune against APIS ground truth.
DEFAULT_FLAG_FRACTION = 0.15

# Below this registration metric quality, mark affected regions "uncertain"
# instead of confidently flagging them. Sign/scale depends on the metric
# used (Mattes MI cost is negative-is-better in SimpleITK) -- calibrate
# this threshold empirically on a handful of known-good vs. known-bad cases.
REGISTRATION_CONFIDENCE_THRESHOLD = None


def region_flags(change_mask, region_labels, flag_fraction=DEFAULT_FLAG_FRACTION):
    """For each ASPECTS region id, compute the fraction of the region
    covered by the change mask and whether it's flagged.

    `region_labels` must be a label volume already in the same array shape
    and voxel grid as `change_mask` (i.e. warped atlas resampled onto the
    patient image, from registration.warp_atlas_labels).
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


def compute_aspects_score(flags):
    """ASPECTS score = 10 - number of flagged regions."""
    flagged_count = sum(1 for f in flags.values() if f["flagged"])
    return max(0, 10 - flagged_count), flagged_count
