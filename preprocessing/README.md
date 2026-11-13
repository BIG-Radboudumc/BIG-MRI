# Preprocessing

Utilities for preparing BIG-MRI DCE breast MRI data. Two versions are provided.

| Version | Script | Summary |
| --- | --- | --- |
| v1 | [`v1/resample_big_mri.py`](v1/resample_big_mri.py) | Resamples each T1 image independently to 1 mm isotropic spacing. |
| v2 | [`v2/preprocess_big_mri.py`](v2/preprocess_big_mri.py) | Standardizes each pre/post-contrast pair: LPS orientation, one shared 1 mm grid, fixed size. |

Both versions expect the same input layout:

```text
<input root>/
  BIGMRI*/              # one folder per patient
    YYYYMMDD/           # one folder per exam
      T1_pre.nii.gz
      T1_post_1.nii.gz
```

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## v1

- Resamples `T1_pre.nii.gz` and every `T1_post*.nii.gz` to `(1.0, 1.0, 1.0)` mm spacing with linear interpolation.
- Keeps the origin and direction of each image. Orientation is not changed.
- Writes 32-bit float NIfTI and mirrors the input folder layout.
- Skips files that already exist in the output directory.

Set `INPUT_ROOT` and `OUTPUT_ROOT` at the top of the script, then run:

```bash
python preprocessing/v1/resample_big_mri.py
```

## v2

For each exam, the pre-contrast image and the first post-contrast image are processed together:

1. Both images are reoriented to LPS.
2. Both images are resampled with B-spline interpolation onto one 1 mm isotropic grid defined by the pre-contrast image.
3. Both images are center-cropped or padded to 352 x 352 x 176 voxels.
4. Both images are saved as 16-bit signed integer NIfTI.

No registration and no intensity normalization are applied. Input files named `T1_W_pre.nii.gz` and `T1_W_post_1.nii.gz` are also accepted. The output is always named `T1_pre.nii.gz` and `T1_post_1.nii.gz` and mirrors the input folder layout. Exams that already exist in the output directory are skipped.

```bash
python preprocessing/v2/preprocess_big_mri.py --input-root /path/to/BIG-MRI --output-root /path/to/output
```
