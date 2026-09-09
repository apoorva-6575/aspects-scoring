"""Phase 2 (Objective 1): symmetry-based ischemic change detection.

No trained model here on purpose -- mirror the volume across the midline,
diff against the original, threshold. This is the guaranteed-working
fallback; refine later with a prompted foundation model if time allows.
"""

import numpy as np
from scipy import ndimage


def brain_center_x(brain_mask):
    """x-centroid of the brain mask -- a much better mirror axis than the
    image's geometric center. On AISD-derived volumes these differ by up
    to ~70px since the head isn't centered in the 512x512 canvas; mirroring
    around the wrong axis silently produces a garbage difference map with
    near-zero overlap with any real lesion (measured: Dice ~0.008 vs
    ~0.015+ after this fix, see scripts/tune_detection.py).
    """
    return np.argwhere(brain_mask)[:, 0].mean()


def mirror_across_x(volume, center_x=None, brain_mask=None, rotation_deg=0.0):
    """Mirror a volume left-right across the brain's true LR symmetry axis.

    If `center_x` isn't given, it's derived from `brain_mask` (see
    `brain_center_x`) rather than defaulting to the image's geometric
    center.

    `rotation_deg` (from preprocessing.find_symmetry_rotation_angle)
    corrects for a tilted head: the volume is rotated to straighten the LR
    axis onto the image x-axis, mirrored there, then rotated back -- so the
    result stays in the original (untilted) coordinate frame and can be
    diffed directly against the input. rotation_deg=0 (the default) skips
    all of this and mirrors along x as before, since most scans don't need
    it and the rotate/mirror/rotate-back path costs more compute.
    """
    if rotation_deg == 0.0:
        if center_x is None:
            center_x = brain_center_x(brain_mask) if brain_mask is not None else (volume.shape[0] - 1) / 2.0
        flipped = volume[::-1, :, :]
        shift = 2 * center_x - (volume.shape[0] - 1)
        return ndimage.shift(flipped, shift=(shift, 0, 0), order=1, mode="nearest")

    # Pad x/y before rotating -- AISD brain masks reach within ~30px of the
    # 512px edge, and rotating with reshape=False clips content at the
    # border asymmetrically depending on angle if there's no margin. This
    # was a real, validated bug (scripts/validate_rotation_correction.py):
    # without padding, the "corrected" mirror was no better than doing
    # nothing, because the rotation itself silently mangled the volume.
    # Right-sized to this specific rotation_deg (not a fixed worst-case
    # assumption) -- padding a full 3D volume is expensive, and most
    # rotation_deg values here are well under the +-20 deg search range
    # find_symmetry_rotation_angle allows.
    pad = int(max(volume.shape[0], volume.shape[1]) / 2 *
              np.sin(np.deg2rad(min(abs(rotation_deg), 45.0)))) + 30
    padded = np.pad(volume, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    straightened = ndimage.rotate(padded, angle=-rotation_deg, axes=(0, 1),
                                   reshape=False, order=1, mode="nearest")
    if center_x is None:
        if brain_mask is None:
            raise ValueError("rotation_deg given but no brain_mask/center_x to find the straightened mirror axis")
        # Recompute the mirror center in the straightened frame -- rotating
        # the volume moves where the brain centroid's x-coordinate is, so
        # reusing the un-rotated center_x here would mirror around the wrong
        # column. Rotate the mask itself (cheap, boolean) rather than
        # hand-deriving the transformed point, to avoid a sign/convention bug.
        padded_mask = np.pad(brain_mask, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        straightened_mask = ndimage.rotate(padded_mask.astype(np.float32), angle=-rotation_deg,
                                           axes=(0, 1), reshape=False, order=0,
                                           mode="constant", cval=0) > 0.5
        center_x = brain_center_x(straightened_mask)
    else:
        center_x = center_x + pad

    flipped = straightened[::-1, :, :]
    shift = 2 * center_x - (straightened.shape[0] - 1)
    mirrored_straight = ndimage.shift(flipped, shift=(shift, 0, 0), order=1, mode="nearest")
    mirrored_padded = ndimage.rotate(mirrored_straight, angle=rotation_deg, axes=(0, 1),
                                     reshape=False, order=1, mode="nearest")
    return mirrored_padded[pad:pad + volume.shape[0], pad:pad + volume.shape[1], :]


def difference_map(volume, mirrored, brain_mask):
    """Voxel-wise |original - mirrored| difference, masked to brain."""
    diff = np.abs(volume - mirrored)
    diff[~brain_mask] = 0
    return diff


def threshold_mask(diff_map, brain_mask, percentile=90, min_blob_voxels=15):
    """Threshold the difference map and drop tiny noise blobs.

    `percentile` is computed only over brain voxels so it adapts per-scan;
    tune this and `min_blob_voxels` against AISD lesion masks.
    """
    brain_values = diff_map[brain_mask]
    if brain_values.size == 0:
        return np.zeros_like(diff_map, dtype=bool)
    thresh = np.percentile(brain_values, percentile)
    mask = diff_map >= thresh

    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    # Vectorized size filter -- thresholding can produce thousands of tiny
    # noise blobs, and a per-blob `labeled == i` loop is O(voxels * blobs).
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    keep_labels = np.nonzero(sizes >= min_blob_voxels)[0] + 1
    return np.isin(labeled, keep_labels)


def detect_ischemic_change(volume, brain_mask, center_x=None, rotation_deg=0.0,
                            percentile=70, min_blob_voxels=80, erode_iterations=3):
    """Full Phase 2 pipeline: returns (diff_map, change_mask).

    Defaults (erode_iterations=3, percentile=70, min_blob_voxels=80) were
    picked by scripts/tune_detection.py, grid-searched against AISD ground
    truth over 60 patients (mean Dice 0.073 -- see README: this is a
    genuinely hard detection problem on NCCT per the problem statement
    itself, and this is the best classical-method result found, not a
    solved detector; min_blob_voxels barely moved the score once erosion
    and percentile were right). `erode_iterations` shrinks the brain mask
    before diffing/thresholding to exclude the skull-strip boundary rim,
    which otherwise dominates the percentile threshold with asymmetric
    edge noise unrelated to any real lesion -- this alone roughly doubled
    Dice in testing.

    `rotation_deg` corrects for a tilted scan (pass
    preprocessing.find_symmetry_rotation_angle's result) -- defaults to 0
    (no correction) since most AISD volumes don't need it and the search +
    rotate/mirror/rotate-back path costs extra compute.
    """
    eroded_mask = (ndimage.binary_erosion(brain_mask, iterations=erode_iterations)
                   if erode_iterations > 0 else brain_mask)
    mirrored = mirror_across_x(volume, center_x, brain_mask=brain_mask, rotation_deg=rotation_deg)
    diff = difference_map(volume, mirrored, eroded_mask)
    mask = threshold_mask(diff, eroded_mask, percentile, min_blob_voxels)
    return diff, mask
