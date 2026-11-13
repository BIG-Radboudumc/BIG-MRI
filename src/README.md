# Baseline

Source code of the BIG-MRI baseline: training, prediction, and evaluation.

The baseline is a 3D ResNet-18. Its input is the subtraction image (first post-contrast minus pre-contrast). It has two outputs: the risk of breast cancer within 2 years and within 5 years. Each output is trained as a binary classification task.

| File | Content |
| --- | --- |
| [`model.py`](model.py) | 3D ResNet-18 with two outputs. Starts from the public MedicalNet weights by default. |
| [`dataset.py`](dataset.py) | Reads the preprocessed images, computes the subtraction image, scales intensities, downsamples to 2 mm, and applies augmentation. |
| [`train.py`](train.py) | Training and validation loop. |
| [`predict.py`](predict.py) | Predicts the 2-year and 5-year risk with a trained model. |
| [`evaluate.py`](evaluate.py) | Computes the AUC and the C-index of the predicted risks. |

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Input

1. Images preprocessed with [`preprocessing/preprocess_images_v2.py`](../preprocessing/).
2. A label table in CSV format with one row per exam, created with [`preprocessing/create_labels.py`](../preprocessing/). The baseline uses these columns:

| Column | Content |
| --- | --- |
| `exam_id` | Exam folder relative to the image folder, for example `BIGMRI_0001/20200101`. |
| `label_2y` | 1 if breast cancer within 2 years, otherwise 0. |
| `label_5y` | 1 if breast cancer within 5 years, otherwise 0. |
| `years_to_cancer` | Time from the exam to breast cancer in years, empty if there is no cancer. Used for the evaluation. |
| `years_to_last_followup` | Follow-up time after the exam in years. Used for the evaluation. |
| `fold` | `test` for the independent test set, or the cross-validation fold `0` to `3`. |

## Training

```bash
python src/train.py --labels /path/to/labels.csv --image-root /path/to/preprocessed --output-dir /path/to/output
```

- The label table assigns every patient to the independent test set or to one of 4 folds. `--fold` selects the validation fold (0 to 3, default 0); the other folds are used for training. The test set is never used during training.
- Positive exams are rare, so they receive a larger weight in the loss.
- The AUC for 2 years and for 5 years is computed on the validation fold after every epoch.

The output folder contains `best_model.pt` (best mean validation AUC), `last_model.pt`, and `log.csv`.

## Prediction

[`predict.py`](predict.py) runs a trained model on every exam in a folder of preprocessed images and writes the predicted risks to a CSV file.

```bash
python src/predict.py --model /path/to/best_model.pt --image-root /path/to/preprocessed --output /path/to/predictions.csv
```

To use an ensemble, pass several models to `--model`. Their predicted risks are averaged, for example the four models of the cross-validation:

```bash
python src/predict.py --model fold0/best_model.pt fold1/best_model.pt fold2/best_model.pt fold3/best_model.pt --image-root /path/to/preprocessed --output /path/to/predictions.csv
```

The output has one row per exam:

| Column | Content |
| --- | --- |
| `exam_id` | Exam folder relative to the image folder, for example `BIGMRI_0001/20200101`. |
| `risk_2y` | Predicted risk of breast cancer within 2 years, between 0 and 1. |
| `risk_5y` | Predicted risk of breast cancer within 5 years, between 0 and 1. |

The 5-year risk is never lower than the 2-year risk: if the model output for 5 years is lower, the 2-year risk is used for both.

## Evaluation

[`evaluate.py`](evaluate.py) compares a prediction table with a label table and reports the AUC for the 2-year and the 5-year risk and Harrell's C-index.

```bash
python src/evaluate.py --predictions /path/to/predictions.csv --labels /path/to/labels.csv --fold test
```

`--fold` selects the part of the label table to evaluate: `test` for the independent test set, or `0` to `3` for one cross-validation fold. Without `--fold`, all exams are evaluated. Evaluate on exams that were not used to train the model.

Add `--output /path/to/metrics.json` to also save the metrics to a file. Only exams that appear in both tables are evaluated.

| Metric | Definition |
| --- | --- |
| `auc_2y` | AUC of `risk_2y` for breast cancer within 2 years. Exams without cancer and with less than 2 years of follow-up are excluded. |
| `auc_5y` | AUC of `risk_5y` for breast cancer within 5 years. Exams without cancer and with less than 5 years of follow-up are excluded. |
| `c_index` | Harrell's C-index of `risk_5y` over the full follow-up. |

See [`docs/evaluation.md`](../docs/evaluation.md) for a detailed explanation of both metrics. The leaderboard computation details are coming soon.
