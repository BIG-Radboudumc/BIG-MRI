from pathlib import Path
from collections import Counter
import math
import SimpleITK as sitk


INPUT_ROOT = Path("/mnt/kronos/data_public/BIG-MRI")
OUTPUT_ROOT = Path("/projects/gan-mri-data/BIG-MRI_preprocessed")

TARGET_SPACING = (1.0, 1.0, 1.0)  # Voxel spacing in mm: x, y, z
OVERWRITE = False


def resample_image(image):
    """Resample a 3D scalar image while preserving its origin and direction."""
    if image.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D image, got {image.GetDimension()} dimensions."
        )

    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar images are supported.")

    old_size = image.GetSize()
    old_spacing = image.GetSpacing()

    if any(not math.isfinite(s) or s <= 0 for s in old_spacing):
        raise ValueError(f"Invalid voxel spacing: {old_spacing}")

    # Cover the physical extent between the original first and last voxel centers.
    new_size = [
        int(math.ceil((n - 1) * s / t)) + 1
        for n, s, t in zip(old_size, old_spacing, TARGET_SPACING)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(image)


def verify_output(path, expected_image):
    """Check the saved image geometry without loading its voxel data."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    if tuple(reader.GetSize()) != tuple(expected_image.GetSize()):
        raise RuntimeError("Output image size verification failed.")

    checks = (
        ("spacing", reader.GetSpacing(), expected_image.GetSpacing()),
        ("origin", reader.GetOrigin(), expected_image.GetOrigin()),
        ("direction", reader.GetDirection(), expected_image.GetDirection()),
    )

    for name, actual, expected in checks:
        if any(
            not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5)
            for a, b in zip(actual, expected)
        ):
            raise RuntimeError(f"Output image {name} verification failed.")


def main():
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_ROOT}")

    input_path = INPUT_ROOT.resolve()
    output_path = OUTPUT_ROOT.resolve()

    if output_path == input_path or input_path in output_path.parents:
        raise ValueError(
            "The output directory must be outside the input directory."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    for patient_dir in sorted(INPUT_ROOT.glob("BIGMRI*")):
        if not patient_dir.is_dir():
            continue

        counts["patients"] += 1

        for study_dir in sorted(patient_dir.iterdir()):
            # Only process date folders named YYYYMMDD.
            if not (
                study_dir.is_dir()
                and len(study_dir.name) == 8
                and study_dir.name.isdigit()
            ):
                continue

            counts["studies"] += 1

            images = sorted(
                path
                for path in study_dir.iterdir()
                if path.is_file()
                and (
                    path.name == "T1_pre.nii.gz"
                    or (
                        path.name.startswith("T1_post")
                        and path.name.endswith(".nii.gz")
                    )
                )
            )

            if not images:
                print(f"[NO MATCHING IMAGES] {study_dir}", flush=True)
                counts["empty_studies"] += 1
                continue

            for source in images:
                counts["found"] += 1
                relative = source.relative_to(INPUT_ROOT)
                destination = OUTPUT_ROOT / relative
                temporary = None

                if destination.exists() and not OVERWRITE:
                    print(f"[SKIPPED: EXISTS] {relative}", flush=True)
                    counts["skipped"] += 1
                    continue

                try:
                    image = sitk.ReadImage(str(source))
                    result = resample_image(image)

                    destination.parent.mkdir(parents=True, exist_ok=True)

                    # Use a unique temporary file in the destination directory.
                    from uuid import uuid4

                    temporary = destination.with_name(
                        f".{uuid4().hex}.tmp.nii.gz"
                    )

                    sitk.WriteImage(result, str(temporary), True)
                    verify_output(temporary, result)

                    # Rename only after writing and verification have completed.
                    temporary.replace(destination)
                    counts["success"] += 1

                    print(
                        f"[DONE] {relative}\n"
                        f"  Size:    {image.GetSize()} -> {result.GetSize()}\n"
                        f"  Spacing: {image.GetSpacing()} -> "
                        f"{result.GetSpacing()}",
                        flush=True,
                    )

                except Exception as exc:
                    counts["failed"] += 1
                    print(f"[FAILED] {relative}: {exc}", flush=True)

                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()

    print("\n========== SUMMARY ==========", flush=True)
    print(f"Patient folders: {counts['patients']}")
    print(f"Study folders: {counts['studies']}")
    print(f"Images found: {counts['found']}")
    print(f"Successfully processed: {counts['success']}")
    print(f"Skipped existing files: {counts['skipped']}")
    print(f"Failed images: {counts['failed']}")
    print(f"Studies without matching images: {counts['empty_studies']}")
    print(f"Output directory: {OUTPUT_ROOT}")

    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
