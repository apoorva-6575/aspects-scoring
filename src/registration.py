"""Phase 3 (Objective 2): register the 2D ASPECTS atlas slices onto the
matching slices of a patient NCCT volume, and warp the region labels onto
them.

The public ASPECTS-281 atlas (figshare.com/articles/figure/ASPECTS-281/
26819290) ships as two 2D reference images per age group -- BGL (basal
ganglia level, labels C=1/L=2/IC=3/I=4/M1=5/M2=6/M3=7) and SGL
(supraganglionic level, labels M4=8/M5=9/M6=10) -- not a 3D volume. This
actually matches real clinical ASPECTS practice: radiologists score from
exactly these two representative axial slices, not the whole brain. So
registration here is per-slice, not per-volume.

No training -- classical optimization-based registration via SimpleITK,
same as a 3D approach would use, just applied in 2D.
"""

import numpy as np
import SimpleITK as sitk


def select_aspects_slices(brain_mask, bg_fraction=0.40, sc_fraction=0.58):
    """Pick the z-slice indices approximating the basal-ganglia level and
    supraganglionic level, as fractions of the brain's z-extent.

    This is a crude placeholder, not real anatomy detection -- clinically,
    BG level is picked by the thalamus/basal ganglia being visible, and SC
    level by seeing the lateral ventricle bodies without basal ganglia.
    Tune bg_fraction/sc_fraction against a handful of AISD volumes with
    known slice anatomy, or swap in real landmark detection if time
    allows -- see README known TODOs.
    """
    z_indices = np.where(brain_mask.any(axis=(0, 1)))[0]
    if z_indices.size == 0:
        raise ValueError("brain_mask is empty -- check skull stripping")
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    z_range = z_max - z_min
    bg_idx = int(round(z_min + bg_fraction * z_range))
    sc_idx = int(round(z_min + sc_fraction * z_range))
    return bg_idx, sc_idx


def _slice_to_sitk(slice_2d):
    # slice_2d is (x, y) per this codebase's convention; sitk.GetImageFromArray
    # treats the first numpy axis as rows, so transpose to keep axes aligned.
    return sitk.GetImageFromArray(slice_2d.T.astype(np.float32))


def register_2d(patient_slice, atlas_image_path):
    """Rigid -> affine registration of a 2D atlas image onto a 2D patient
    slice. Returns (transform, metric_value) -- metric is a rough proxy
    for registration confidence (lower Mattes MI cost is usually better;
    calibrate a threshold empirically, don't trust the raw number blindly).
    """
    fixed = _slice_to_sitk(patient_slice)
    moving = sitk.ReadImage(atlas_image_path, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetInitialTransform(initial, inPlace=False)
    rigid_transform = rigid_reg.Execute(fixed, moving)

    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetInitialTransform(rigid_transform, inPlace=False)
    affine_transform = affine_reg.Execute(fixed, moving)
    metric_value = affine_reg.GetMetricValue()

    return affine_transform, metric_value


def warp_2d_labels(patient_slice, atlas_label_path, transform):
    """Warp a 2D atlas label image onto the patient slice grid (nearest
    neighbor, to keep integer labels intact). Returns an (x, y) array
    matching `patient_slice`'s shape/orientation.
    """
    fixed = _slice_to_sitk(patient_slice)
    labels = sitk.ReadImage(atlas_label_path, sitk.sitkUInt8)
    warped = sitk.Resample(labels, fixed, transform, sitk.sitkNearestNeighbor,
                           0, labels.GetPixelID())
    return sitk.GetArrayFromImage(warped).T


def register_aspects_atlas(patient_windowed, brain_mask,
                            bgl_image_path, bgl_label_path,
                            sgl_image_path, sgl_label_path,
                            bg_fraction=0.40, sc_fraction=0.58):
    """Full Objective 2 pipeline: pick the BG/SC slices out of the patient
    volume, register both atlas levels onto them, and return the warped
    region-label maps plus which patient slice each came from.
    """
    bg_idx, sc_idx = select_aspects_slices(brain_mask, bg_fraction, sc_fraction)
    bg_slice = patient_windowed[:, :, bg_idx]
    sc_slice = patient_windowed[:, :, sc_idx]

    bg_transform, bg_metric = register_2d(bg_slice, bgl_image_path)
    bg_region_labels = warp_2d_labels(bg_slice, bgl_label_path, bg_transform)

    sc_transform, sc_metric = register_2d(sc_slice, sgl_image_path)
    sc_region_labels = warp_2d_labels(sc_slice, sgl_label_path, sc_transform)

    return {
        "bg_slice_idx": bg_idx,
        "sc_slice_idx": sc_idx,
        "bg_region_labels": bg_region_labels,
        "sc_region_labels": sc_region_labels,
        "bg_metric": bg_metric,
        "sc_metric": sc_metric,
    }
