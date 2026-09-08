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


def mirror_across_x(volume, center_x=None, brain_mask=None):
    """Mirror a volume left-right across a given x index.

    Assumes the volume is already roughly axis-aligned (x = left-right),
    which holds for most clinical NCCT after standard loading. If scans in
    your dataset are tilted, rotate to align `lr_axis` from preprocessing
    with the x-axis before calling this.

    If `center_x` isn't given, it's derived from `brain_mask` (see
    `brain_center_x`) rather than defaulting to the image's geometric
    center -- pass `brain_mask` explicitly, don't rely on the old
    geometric-center fallback.
    """
    if center_x is None:
        if brain_mask is None:
            center_x = (volume.shape[0] - 1) / 2.0
        else:
            center_x = brain_center_x(brain_mask)
    flipped = volume[::-1, :, :]
    shift = 2 * center_x - (volume.shape[0] - 1)
    mirrored = ndimage.shift(flipped, shift=(shift, 0, 0), order=1, mode="nearest")
    return mirrored


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


def detect_ischemic_change(volume, brain_mask, center_x=None,
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
    """
    eroded_mask = (ndimage.binary_erosion(brain_mask, iterations=erode_iterations)
                   if erode_iterations > 0 else brain_mask)
    mirrored = mirror_across_x(volume, center_x, brain_mask=brain_mask)
    diff = difference_map(volume, mirrored, eroded_mask)
    mask = threshold_mask(diff, eroded_mask, percentile, min_blob_voxels)
    return diff, mask
