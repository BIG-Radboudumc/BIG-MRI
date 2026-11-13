# Preprocessing

Utilities for preparing BIG-MRI DCE breast MRI data. Two versions are provided.

| Version | Script | Summary |
| --- | --- | --- |
| v1 | [`preprocess_images_v1.py`](preprocess_images_v1.py) | Resamples each T1 image independently to 1 mm isotropic spacing. |
| v2 | [`preprocess_images_v2.py`](preprocess_images_v2.py) | Standardizes each pre/post-contrast pair: LPS orientation, one shared 1 mm grid, fixed size. |

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
python preprocessing/preprocess_images_v1.py
```

## v2

For each exam, the pre-contrast image and the first post-contrast image are processed together:

1. Both images are reoriented to LPS.
2. Both images are resampled with B-spline interpolation onto one 1 mm isotropic grid defined by the pre-contrast image.
3. Both images are center-cropped or padded to 352 x 352 x 176 voxels.
4. Both images are saved as 16-bit signed integer NIfTI.

No registration and no intensity normalization are applied. Input files named `T1_W_pre.nii.gz` and `T1_W_post_1.nii.gz` are also accepted. The output is always named `T1_pre.nii.gz` and `T1_post_1.nii.gz` and mirrors the input folder layout. Exams that already exist in the output directory are skipped.

```bash
python preprocessing/preprocess_images_v2.py --input-root /path/to/BIG-MRI --output-root /path/to/output
```

## Label table

[`create_labels.py`](create_labels.py) creates the label table of the baseline from the clinical table. The clinical table has one row per lesion; the label table has one row per exam.

```bash
python preprocessing/create_labels.py --clinical /path/to/clinical.xlsx --output /path/to/labels.csv
```

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier (`release_id` in the clinical table). |
| `exam_id` | Exam folder (`exam_folder` in the clinical table), for example `BIGMRI_0001/20200101`. |
| `label_2y` | 1 if breast cancer within 2 years after the exam, otherwise 0. |
| `label_5y` | 1 if breast cancer within 5 years after the exam, otherwise 0. |
| `years_to_cancer` | Time from the exam to breast cancer in years. Empty if there is no cancer. |
| `years_to_last_followup` | Follow-up time after the exam in years. |
| `fold` | `test` for the independent test set, or the cross-validation fold `0` to `3`. |

How the columns are defined:

- **Time to cancer.** If `cancer_present` is 1, the time is 0. Otherwise, if the cancer detection date is after the exam date, the time is the interval between the two dates. In all other cases (no detection date, or a detection date before the exam) the exam has no cancer and the time is empty. No exam is excluded.
- **Labels.** `label_2y` is 1 if the time to cancer is less than 2 years, and `label_5y` is 1 if it is less than 5 years. All other exams are labeled 0, regardless of their follow-up time.
- **Follow-up.** The follow-up of a patient is assumed to end one year after the last exam of that patient. The follow-up time of an exam is the interval from the exam to that date.
- **Data split.** 20% of the patients form the independent test set. The other patients are divided into 4 cross-validation folds. The split is made by patient, so all exams of a patient are in the same part, and every part has a similar share of patients with cancer. The split is fixed by `--seed` (default 0).
