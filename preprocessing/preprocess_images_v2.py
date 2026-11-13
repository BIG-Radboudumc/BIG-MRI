"""Preprocess BIG-MRI T1 pre-contrast and first post-contrast images.

Every exam is brought to the same standard form:

1. Reorient both images to LPS.
2. Resample both images onto one 1 mm isotropic grid defined by the
   pre-contrast image (B-spline interpolation).
3. Center-crop or pad to 352 x 352 x 176 voxels.
4. Save as 16-bit signed integer NIfTI.

No registration and no intensity normalization are applied.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


TARGET_SPACING = (1.0, 1.0, 1.0)  # Voxel spacing in mm: x, y, z
TARGET_SIZE = (352, 352, 176)  # Image size in voxels: x, y, z

# Accepted input file names (matched case-insensitively): pre, first post.
INPUT_NAMES = (
    ("t1_pre.nii.gz", "t1_post_1.nii.gz"),
    ("t1_w_pre.nii.gz", "t1_w_post_1.nii.gz"),
)
OUTPUT_NAMES = ("T1_pre.nii.gz", "T1_post_1.nii.gz")


def resample(image, grid):
    """Resample an image onto the given grid with B-spline interpolation."""
    # Outside the image, use its corner value as background.
    background = float(image.GetPixel(0, 0, 0))
    resampled = sitk.Resample(
        image,
        grid,
        sitk.Transform(),
        sitk.sitkBSpline,
        background,
        sitk.sitkFloat64,
    )

    # Convert back to the pixel type of the input image.
    pixel_type = sitk.GetArrayViewFromImage(image).dtype
    if np.issubdtype(pixel_type, np.integer):
        limits = np.iinfo(pixel_type)
        return sitk.Clamp(
            resampled,
            image.GetPixelID(),
            float(limits.min),
            float(limits.max),
        )
    return sitk.Cast(resampled, image.GetPixelID())


def center_crop_or_pad(image):
    """Center-crop or pad an image to TARGET_SIZE, keeping physical coordinates."""
    target = np.array(TARGET_SIZE)

    # Pad with the corner value where the image is smaller than the target.
    padding = np.maximum(target - image.GetSize(), 0)
    lower = padding // 2
    upper = padding - lower
    image = sitk.ConstantPad(
        image, lower.tolist(), upper.tolist(), float(image.GetPixel(0, 0, 0))
    )

    # Crop where the image is larger than the target.
    start = np.maximum(np.array(image.GetSize()) - target, 0) // 2
    return sitk.RegionOfInterest(image, list(TARGET_SIZE), start.tolist())


def to_int16(image):
    """Convert an image to 16-bit signed integers."""
    values = sitk.GetArrayViewFromImage(image)
    if values.min() < -32768 or values.max() > 32767:
        raise ValueError("Image intensities do not fit in 16-bit signed integers.")
    return sitk.Cast(image, sitk.sitkInt16)


def preprocess_pair(pre, post):
    """Return the standardized (pre, post) images on one shared grid."""
    # 1. Reorient to LPS. Physical coordinates are preserved.
    pre = sitk.DICOMOrient(pre, "LPS")
    post = sitk.DICOMOrient(post, "LPS")

    # 2. Define the 1 mm grid from the pre-contrast image.
    grid_size = np.ceil(
        np.array(pre.GetSize()) * pre.GetSpacing() / np.array(TARGET_SPACING)
    )
    grid = sitk.Image([int(n) for n in grid_size], sitk.sitkUInt8)
    grid.SetSpacing(TARGET_SPACING)
    grid.SetOrigin(pre.GetOrigin())
    grid.SetDirection(pre.GetDirection())

    # 3. Resample both images onto that grid, then crop or pad, then cast.
    pre = to_int16(center_crop_or_pad(resample(pre, grid)))
    post = to_int16(center_crop_or_pad(resample(post, grid)))
    return pre, post


def find_pairs(input_root):
    """Return (pre, post) file paths for every exam folder under input_root."""
    pairs = []
    for exam_dir in sorted(input_root.glob("BIGMRI*/*")):
        if not exam_dir.is_dir():
            continue
        files = {path.name.lower(): path for path in exam_dir.iterdir()}
        for pre_name, post_name in INPUT_NAMES:
            if pre_name in files and post_name in files:
                pairs.append((files[pre_name], files[post_name]))
                break
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="BIG-MRI dataset folder"
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="Folder for the output"
    )
    args = parser.parse_args()

    pairs = find_pairs(args.input_root)
    print(f"Found {len(pairs)} exams.", flush=True)
    failed = 0

    for index, (pre_path, post_path) in enumerate(pairs, start=1):
        exam = pre_path.parent.relative_to(args.input_root)
        output_dir = args.output_root / exam
        pre_output = output_dir / OUTPUT_NAMES[0]
        post_output = output_dir / OUTPUT_NAMES[1]

        # Skip exams that are already done, so a run can be resumed.
        if pre_output.exists() and post_output.exists():
            print(f"[{index}/{len(pairs)}] SKIPPED: EXISTS {exam}", flush=True)
            continue

        try:
            pre, post = preprocess_pair(
                sitk.ReadImage(str(pre_path)), sitk.ReadImage(str(post_path))
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(pre, str(pre_output), True)
            sitk.WriteImage(post, str(post_output), True)
            print(f"[{index}/{len(pairs)}] DONE {exam}", flush=True)
        except Exception as error:
            failed += 1
            print(f"[{index}/{len(pairs)}] FAILED {exam}: {error}", flush=True)

    print(f"Finished. Failed exams: {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
