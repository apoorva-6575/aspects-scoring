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


def find_midline_axis(brain_mask):
    """Estimate the left-right symmetry axis via PCA on the brain mask.

    Returns (centroid, direction_vector) in voxel coordinates, where
    direction_vector is the axis to mirror across (approx. anterior-posterior
    axis in axial slices; the mirror plane is perpendicular to the
    left-right axis found here).
    """
    coords = np.argwhere(brain_mask)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # left-right axis is typically the component with largest variance
    # in the x (left-right) image dimension -- verify per-dataset orientation.
    lr_axis = eigvecs[:, np.argmax(eigvals)]
    return centroid, lr_axis


def preprocess_volume(path, hu_center=40, hu_width=80):
    """Convenience wrapper for the full Phase 1 pipeline on one file."""
    raw, affine, img = load_nifti(path)
    windowed = window_hu(raw, hu_center, hu_width)
    brain_mask = skull_strip_threshold(raw)
    centroid, lr_axis = find_midline_axis(brain_mask)
    return {
        "raw": raw,
        "windowed": windowed,
        "brain_mask": brain_mask,
        "centroid": centroid,
        "lr_axis": lr_axis,
        "affine": affine,
    }
