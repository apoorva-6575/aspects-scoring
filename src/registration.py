"""Phase 3 (Objective 2): register the NCCT ASPECTS atlas onto a patient
scan and warp the 10 region labels into patient space. No training --
classical optimization-based registration via SimpleITK.
"""

import SimpleITK as sitk


def _read(path):
    return sitk.ReadImage(path, sitk.sitkFloat32)


def register_atlas_to_patient(atlas_image_path, patient_image_path):
    """Rigid -> affine -> BSpline deformable registration.

    Returns (composite_transform, registration_metric_value). The metric
    value (final mutual-information cost) is a rough proxy for registration
    confidence -- track it per-case and flag low-confidence registrations
    downstream instead of silently trusting the warp.
    """
    fixed = _read(patient_image_path)   # patient = fixed
    moving = _read(atlas_image_path)    # atlas = moving

    # 1. Rigid initialization
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetInitialTransform(initial, inPlace=False)
    rigid_transform = rigid_reg.Execute(fixed, moving)

    # 2. Affine refinement, initialized from the rigid result
    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetInitialTransform(rigid_transform, inPlace=False)
    affine_transform = affine_reg.Execute(fixed, moving)

    # 3. BSpline deformable refinement on top of the affine result
    mesh_size = [8] * fixed.GetDimension()
    bspline = sitk.BSplineTransformInitializer(fixed, mesh_size)

    deformable_reg = sitk.ImageRegistrationMethod()
    deformable_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    deformable_reg.SetOptimizerAsLBFGSB(numberOfIterations=100)
    deformable_reg.SetInterpolator(sitk.sitkLinear)
    deformable_reg.SetMovingInitialTransform(affine_transform)
    deformable_reg.SetInitialTransform(bspline, inPlace=False)
    deformable_transform = deformable_reg.Execute(fixed, moving)
    metric_value = deformable_reg.GetMetricValue()

    composite = sitk.CompositeTransform([affine_transform, deformable_transform])
    return composite, metric_value


def warp_atlas_labels(atlas_labels_path, patient_image_path, transform):
    """Warp the atlas's discrete region-label image into patient space.
    Uses nearest-neighbor interpolation to keep labels as integers.
    """
    fixed = _read(patient_image_path)
    labels = sitk.ReadImage(atlas_labels_path, sitk.sitkUInt8)

    warped = sitk.Resample(
        labels, fixed, transform, sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
    return sitk.GetArrayFromImage(warped)  # z, y, x -- watch axis order vs. nibabel
