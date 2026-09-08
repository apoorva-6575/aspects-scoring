"""Overlay helpers for the Streamlit app: change-mask heatmap + region
boundary contours on a single axial slice.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def render_slice(ct_slice, change_mask_slice=None, region_labels_slice=None, title=""):
    """Return a matplotlib Figure with CT + optional overlays for one slice."""
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(ct_slice.T, cmap="gray", origin="lower")

    if change_mask_slice is not None and change_mask_slice.any():
        heat = np.ma.masked_where(~change_mask_slice.T, change_mask_slice.T)
        ax.imshow(heat, cmap="autumn", alpha=0.5, origin="lower")

    if region_labels_slice is not None:
        ax.contour(region_labels_slice.T, levels=np.arange(0.5, 10.5, 1),
                   colors="cyan", linewidths=0.5, origin="lower")

    ax.set_title(title)
    ax.axis("off")
    return fig
