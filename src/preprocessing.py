"""Phase 1: load NCCT, window to stroke HU range, skull-strip, find midline."""

import numpy as np
import nibabel as nib
from scipy import ndimage


def load_nifti(path):
    """Return (volume: np.ndarray [x,y,z], affine, nib image)."""
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    return vol, img.affine, img


def window_hu(volume, center=40, width=80):
    """Apply a stroke/soft-tissue HU window and rescale to 0-1."""
    low, high = center - width / 2, center + width / 2
    windowed = np.clip(volume, low, high)
    return (windowed - low) / (high - low)


def skull_strip_threshold(volume, hu_low=0, hu_high=100):
    """Crude fallback skull strip: HU threshold + largest connected component
    + binary fill holes. Swap this out for HD-BET if time allows — this is
    only meant to unblock the rest of the pipeline early.
    """
    mask = (volume >= hu_low) & (volume <= hu_high)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    brain_mask = labeled == largest
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = ndimage.binary_closing(brain_mask, iterations=2)
    return brain_mask


def find_symmetry_rotation_angle(windowed, brain_mask, angle_range=20.0, angle_step=1.0,
                                  mask_erosion=8):
    """Search for the in-plane rotation (degrees) that best aligns the
    brain's true left-right symmetry axis with the image x-axis.

    Two earlier approaches were tried and empirically failed (see
    scripts/validate_rotation_correction.py, which injects a known
    synthetic tilt and checks it's recovered):

    1. A PCA-based approach (the original find_midline_axis) that inferred
       the LR axis from shape variance -- unreliable since which principal
       axis is LR vs. AP depends on assuming one is reliably longer, and it
       was never actually wired into detection.
    2. Maximizing bilateral overlap of the boolean skull-strip *silhouette*
       against its own mirror. This seemed reasonable but the measured
       score curve across candidate angles was monotonic with no interior
       peak -- it kept "improving" all the way to the search boundary. The
       skull-strip mask is close to an oval, and an oval is trivially near-
       symmetric about many axes, so the silhouette alone carries almost no
       signal about true head orientation; what little shape irregularity
       exists (imperfect skull-stripping, or -- since this is a stroke
       dataset -- real pathological mass effect) dominated the score
       instead, unrelated to actual tilt.

    This version instead correlates *intensity content* (not just outer
    shape) between a candidate-angle rotation and its mirror, restricted to
    an eroded brain region (excludes the skull-strip boundary rim, which is
    noisy and asymmetric). Real anatomy -- ventricles, gray/white matter
    boundaries, sulcal pattern -- gives a far more specific bilateral signal
    than an outer contour. Validated: recovers injected synthetic tilts of
    +8 deg and +12 deg to within 1 deg.

    Runs on the single largest-area axial slice (not a union across
    slices -- different anatomical levels have different shapes) and pads
    with margin before rotating, since `reshape=False` rotation clips
    content at the array edges asymmetrically depending on angle whenever
    the mask extends close to the border (it does on AISD: masks reach
    within ~30px of the 512px edge).
    """
    areas = brain_mask.sum(axis=(0, 1))
    if not areas.any():
        return 0.0, 0.0
    z = int(np.argmax(areas))
    img_slice = windowed[:, :, z]
    mask_slice = ndimage.binary_erosion(brain_mask[:, :, z], iterations=mask_erosion)
    # Pad enough that rotating by up to angle_range doesn't clip content at
    # the edge, without padding so generously that rotation gets slow --
    # worst-case displacement of a corner at max radius is
    # radius * sin(angle_range), plus a safety margin.
    pad = int(max(img_slice.shape) / 2 * np.sin(np.deg2rad(angle_range))) + 30
    img_padded = np.pad(img_slice, pad, mode="constant")
    mask_padded = np.pad(mask_slice, pad, mode="constant", constant_values=False)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-angle_range, angle_range + 1e-9, angle_step):
        rotated_img = ndimage.rotate(img_padded, angle=-angle, reshape=False,
                                     order=1, mode="nearest")
        rotated_mask = ndimage.rotate(mask_padded.astype(np.float32), angle=-angle,
                                      reshape=False, order=0, mode="constant", cval=0) > 0.5
        if not rotated_mask.any():
            continue
        center_x = np.argwhere(rotated_mask)[:, 0].mean()
        shift = 2 * center_x - (rotated_img.shape[0] - 1)
        mirrored_img = ndimage.shift(rotated_img[::-1, :], shift=(shift, 0), order=1, mode="nearest")
        mirrored_mask = ndimage.shift(rotated_mask[::-1, :].astype(np.float32), shift=(shift, 0),
                                      order=0, mode="constant", cval=0) > 0.5

        overlap = rotated_mask & mirrored_mask
        if overlap.sum() < 100:
            continue
        a, b = rotated_img[overlap], mirrored_img[overlap]
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        score = float(np.corrcoef(a, b)[0, 1])
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle, best_score


def preprocess_volume(path, hu_center=40, hu_width=80, already_windowed=False,
                       ss_low=None, ss_high=None):
    """Convenience wrapper for the full Phase 1 pipeline on one file.

    `already_windowed=True` is for volumes converted from AISD's PNG slices
    (see scripts/convert_aisd_to_nifti.py): those are exported as 0-255
    display-windowed images, not raw Hounsfield units, so applying
    `window_hu` again would be wrong. In that mode we just rescale to 0-1
    and use 0-255-scale skull-strip thresholds instead of HU thresholds.
    """
    raw, affine, img = load_nifti(path)
    if already_windowed:
        windowed = np.clip(raw, 0, 255) / 255.0
        low = 20 if ss_low is None else ss_low
        high = 230 if ss_high is None else ss_high
    else:
        windowed = window_hu(raw, hu_center, hu_width)
        low = 0 if ss_low is None else ss_low
        high = 100 if ss_high is None else ss_high

    brain_mask = skull_strip_threshold(raw, low, high)
    rotation_deg, rotation_score = find_symmetry_rotation_angle(windowed, brain_mask)
    return {
        "raw": raw,
        "windowed": windowed,
        "brain_mask": brain_mask,
        "rotation_deg": rotation_deg,
        "rotation_symmetry_score": rotation_score,
        "affine": affine,
    }
