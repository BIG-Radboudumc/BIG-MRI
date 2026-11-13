# Training

Baseline training pipeline for BIG-MRI.

The baseline is a 3D ResNet-18. Its input is the subtraction image (first post-contrast minus pre-contrast). It has two outputs: the risk of breast cancer within 2 years and within 5 years. Each output is trained as a binary classification task.

| File | Content |
| --- | --- |
| [`model.py`](model.py) | 3D ResNet-18 with two outputs. Starts from the public MedicalNet weights by default. |
| [`dataset.py`](dataset.py) | Reads the preprocessed images, computes the subtraction image, scales intensities, downsamples to 2 mm, and applies augmentation. |
| [`train.py`](train.py) | Training and validation loop. |

## Input

1. Images preprocessed with [`preprocessing/v2`](../preprocessing/).
2. A label table in CSV format with one row per exam:

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier, used to split the data by patient. |
| `exam_id` | Exam folder relative to the image folder, for example `BIGMRI_0001/20200101`. |
| `label_2y` | 1 if breast cancer within 2 years, otherwise 0. |
| `label_5y` | 1 if breast cancer within 5 years, otherwise 0. |

The script that creates this table from the clinical data is coming soon.

## Usage

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python training/train.py --labels /path/to/labels.csv --image-root /path/to/preprocessed --output-dir /path/to/output
```

- The data is split into 5 folds by patient. `--fold` selects the validation fold (default 0).
- Positive exams are rare, so they receive a larger weight in the loss.
- The AUC for 2 years and for 5 years is computed on the validation fold after every epoch.

The output folder contains `best_model.pt` (best mean validation AUC), `last_model.pt`, and `log.csv`.
