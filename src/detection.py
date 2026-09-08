"""Phase 2 (Objective 1): symmetry-based ischemic change detection.

No trained model here on purpose -- mirror the volume across the midline,
diff against the original, threshold. This is the guaranteed-working
fallback; refine later with a prompted foundation model if time allows.
"""

import numpy as np
from scipy import ndimage


def mirror_across_x(volume, center_x=None):
    """Mirror a volume left-right across a given x index.

    Assumes the volume is already roughly axis-aligned (x = left-right),
    which holds for most clinical NCCT after standard loading. If scans in
    your dataset are tilted, rotate to align `lr_axis` from preprocessing
    with the x-axis before calling this.
    """
    if center_x is None:
        center_x = (volume.shape[0] - 1) / 2.0
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
                            percentile=90, min_blob_voxels=15):
    """Full Phase 2 pipeline: returns (diff_map, change_mask)."""
    mirrored = mirror_across_x(volume, center_x)
    diff = difference_map(volume, mirrored, brain_mask)
    mask = threshold_mask(diff, brain_mask, percentile, min_blob_voxels)
    return diff, mask
