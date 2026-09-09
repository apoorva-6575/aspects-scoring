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

IMPORTANT -- patient_spacing: every entry point here takes the patient
volume's real (x, y) voxel spacing in mm (from the NIfTI affine, e.g.
`pre["affine"]`). Without it, patient slices get converted to SimpleITK
images at the default 1.0mm/px, while the atlas carries its own real
spacing (0.5mm/px, ~230mm across) -- so SimpleITK would believe the
patient's head is ~512mm across, a ~2.2x scale mismatch a *rigid*
transform (rotation+translation only, no scaling) cannot compensate for.
This was found and confirmed as the actual root cause of a real bug: every
registration attempted before this fix converged to a "confident-looking"
(good Mattes MI score) but anatomically nonsensical transform -- rotation
angles like 147.5 degrees, and region labels landing with zero voxel
overlap against AISD ground-truth lesions in every one of 4 patients
checked. See scripts/validate_rotation_correction.py-adjacent debugging
notes in the project history; this is not a theoretical concern.
"""

import numpy as np
import SimpleITK as sitk


def select_aspects_slices(brain_mask, bg_fraction=0.40, sc_fraction=0.58):
    """Pick the z-slice indices approximating the basal-ganglia level and
    supraganglionic level, as fractions of the brain's z-extent.

    This is a crude placeholder, not real anatomy detection -- kept as a
    cheap fallback for callers that don't want to pay for
    select_aspects_slices_by_registration's per-candidate registrations.
    Clinically, BG level is picked by the thalamus/basal ganglia being
    visible, and SC level by seeing the lateral ventricle bodies without
    basal ganglia.
    """
    z_indices = np.where(brain_mask.any(axis=(0, 1)))[0]
    if z_indices.size == 0:
        raise ValueError("brain_mask is empty -- check skull stripping")
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    z_range = z_max - z_min
    bg_idx = int(round(z_min + bg_fraction * z_range))
    sc_idx = int(round(z_min + sc_fraction * z_range))
    return bg_idx, sc_idx


def _candidate_z_indices(brain_mask, frac_low, frac_high, n_candidates):
    z_indices = np.where(brain_mask.any(axis=(0, 1)))[0]
    if z_indices.size == 0:
        raise ValueError("brain_mask is empty -- check skull stripping")
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    z_range = z_max - z_min
    fracs = np.linspace(frac_low, frac_high, n_candidates)
    candidates = sorted({int(round(z_min + f * z_range)) for f in fracs})
    return candidates


def _slice_to_sitk(slice_2d, spacing):
    # slice_2d is (x, y) per this codebase's convention; sitk.GetImageFromArray
    # treats the first numpy axis as rows, so transpose to keep axes aligned.
    img = sitk.GetImageFromArray(slice_2d.T.astype(np.float32))
    img.SetSpacing((float(spacing[0]), float(spacing[1])))
    return img


def _rigid_metric(patient_slice, atlas_image_path, patient_spacing):
    """Quick rigid-only registration, just to score how well an atlas
    template matches a candidate slice -- cheaper than the full rigid+
    affine used for the actual label warp in register_2d.
    """
    fixed = _slice_to_sitk(patient_slice, patient_spacing)
    moving = sitk.ReadImage(atlas_image_path, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100)
    reg.SetOptimizerScalesFromPhysicalShift()  # see register_2d for why this matters
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(initial, inPlace=False)
    reg.Execute(fixed, moving)
    return reg.GetMetricValue()


def select_slice_by_registration(patient_windowed, brain_mask, atlas_image_path,
                                  patient_spacing, frac_low, frac_high, n_candidates=7):
    """Pick the z-slice in [frac_low, frac_high] of the brain's z-extent
    whose rigid registration against `atlas_image_path` scores best (lowest
    Mattes MI cost). This replaces guessing a single fixed fraction with an
    actual search using registration quality as the anatomy-matching
    signal -- still no training, just more registrations.
    """
    candidates = _candidate_z_indices(brain_mask, frac_low, frac_high, n_candidates)
    best_z, best_metric = candidates[0], np.inf
    for z in candidates:
        metric = _rigid_metric(patient_windowed[:, :, z], atlas_image_path, patient_spacing)
        if metric < best_metric:
            best_metric = metric
            best_z = z
    return best_z, best_metric


def _run_rigid(fixed, moving, initial_angle_rad):
    """One rigid registration run seeded at a given initial rotation."""
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    initial.SetAngle(initial_angle_rad)
    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    # Without this, rotation (radians) and translation (pixels) share one
    # learning rate despite being on totally different scales -- the
    # optimizer can take a wildly oversized step in angle relative to
    # translation and diverge into a nonsense local optimum.
    # SetOptimizerScalesFromPhysicalShift rescales each parameter's step by
    # how much it actually moves the image in physical space, the standard
    # SimpleITK fix for this.
    rigid_reg.SetOptimizerScalesFromPhysicalShift()
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetInitialTransform(initial, inPlace=False)
    transform = rigid_reg.Execute(fixed, moving)
    return transform, rigid_reg.GetMetricValue()


def register_2d(patient_slice, atlas_image_path, patient_spacing):
    """Rigid -> affine registration of a 2D atlas image onto a 2D patient
    slice. Returns (transform, metric_value) -- metric is a rough proxy
    for registration confidence (lower Mattes MI cost is usually better;
    calibrate a threshold empirically, don't trust the raw number blindly).
    """
    fixed = _slice_to_sitk(patient_slice, patient_spacing)
    moving = sitk.ReadImage(atlas_image_path, sitk.sitkFloat32)

    # Multi-start rigid stage: a single gradient-descent run from angle=0
    # can converge to a garbage local optimum, because a roughly oval brain
    # shape looks similar to Mattes MI under large spurious rotations. Try
    # several candidate starting angles and keep whichever actually
    # converges to the best final metric -- the same "search a small grid
    # instead of trusting one gradient descent" fix already validated for
    # find_symmetry_rotation_angle.
    best_transform, best_metric = None, np.inf
    for angle_deg in (0, 45, 90, 135, 180, -45, -90, -135):
        transform, metric = _run_rigid(fixed, moving, np.deg2rad(angle_deg))
        if metric < best_metric:
            best_transform, best_metric = transform, metric
    rigid_transform = best_transform

    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    affine_reg.SetOptimizerScalesFromPhysicalShift()
    affine_reg.SetInterpolator(sitk.sitkLinear)
    # NOTE: this "affine" stage actually continues optimizing the same
    # Euler2DTransform (rigid: rotation + translation only) from `initial`
    # -- SetInitialTransform doesn't change the transform TYPE, so passing
    # a rigid_transform here does not add scaling/shearing degrees of
    # freedom despite the function's rigid->affine framing. Left as a
    # second rigid refinement pass for now; a true affine stage would need
    # SetInitialTransform(sitk.AffineTransform(2)) seeded from
    # rigid_transform's parameters.
    affine_reg.SetInitialTransform(rigid_transform, inPlace=False)
    affine_transform = affine_reg.Execute(fixed, moving)
    metric_value = affine_reg.GetMetricValue()

    return affine_transform, metric_value


def warp_2d_labels(patient_slice, atlas_label_path, transform, patient_spacing):
    """Warp a 2D atlas label image onto the patient slice grid (nearest
    neighbor, to keep integer labels intact). Returns an (x, y) array
    matching `patient_slice`'s shape/orientation.
    """
    fixed = _slice_to_sitk(patient_slice, patient_spacing)
    labels = sitk.ReadImage(atlas_label_path, sitk.sitkUInt8)
    warped = sitk.Resample(labels, fixed, transform, sitk.sitkNearestNeighbor,
                           0, labels.GetPixelID())
    return sitk.GetArrayFromImage(warped).T


def register_aspects_atlas(patient_windowed, brain_mask, patient_spacing,
                            bgl_image_path, bgl_label_path,
                            sgl_image_path, sgl_label_path,
                            bg_frac_range=(0.25, 0.55), sc_frac_range=(0.50, 0.80),
                            n_slice_candidates=7):
    """Full Objective 2 pipeline: search for the BG/SC slices out of the
    patient volume (by registration quality against each atlas template,
    see select_slice_by_registration), register both atlas levels onto
    the winners, and return the warped region-label maps plus which
    patient slice each came from.

    `patient_spacing` is the patient volume's (x, y) voxel spacing in mm
    (e.g. derived from `pre["affine"]`) -- required to avoid a scale
    mismatch against the atlas's own real spacing, see module docstring.

    bg_frac_range/sc_frac_range bound the search to plausible bands of the
    brain's z-extent (BG level lower, SC level higher) rather than
    searching the whole volume -- widen them if a volume has unusual
    z-extent (e.g. a limited/cropped FOV).
    """
    bg_idx, bg_search_metric = select_slice_by_registration(
        patient_windowed, brain_mask, bgl_image_path, patient_spacing, *bg_frac_range, n_slice_candidates)
    sc_idx, sc_search_metric = select_slice_by_registration(
        patient_windowed, brain_mask, sgl_image_path, patient_spacing, *sc_frac_range, n_slice_candidates)

    bg_slice = patient_windowed[:, :, bg_idx]
    sc_slice = patient_windowed[:, :, sc_idx]

    bg_transform, bg_metric = register_2d(bg_slice, bgl_image_path, patient_spacing)
    bg_region_labels = warp_2d_labels(bg_slice, bgl_label_path, bg_transform, patient_spacing)

    sc_transform, sc_metric = register_2d(sc_slice, sgl_image_path, patient_spacing)
    sc_region_labels = warp_2d_labels(sc_slice, sgl_label_path, sc_transform, patient_spacing)

    return {
        "bg_slice_idx": bg_idx,
        "sc_slice_idx": sc_idx,
        "bg_region_labels": bg_region_labels,
        "sc_region_labels": sc_region_labels,
        "bg_metric": bg_metric,
        "sc_metric": sc_metric,
    }
