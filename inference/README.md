# Inference

Baseline inference and testing scripts for BIG-MRI.

[`predict.py`](predict.py) runs a trained baseline model on every exam in a folder and writes the predicted 2-year and 5-year risk to a CSV file.

## Input

1. A model trained with [`training/train.py`](../training/), for example `best_model.pt`.
2. Images preprocessed with [`preprocessing/v2`](../preprocessing/).

## Usage

```bash
python inference/predict.py --model /path/to/best_model.pt --image-root /path/to/preprocessed --output /path/to/predictions.csv
```

## Output

A CSV file with one row per exam:

| Column | Content |
| --- | --- |
| `exam_id` | Exam folder relative to the image folder, for example `BIGMRI_0001/20200101`. |
| `risk_2y` | Predicted risk of breast cancer within 2 years, between 0 and 1. |
| `risk_5y` | Predicted risk of breast cancer within 5 years, between 0 and 1. |

The 5-year risk is never lower than the 2-year risk: if the model output for 5 years is lower, the 2-year risk is used for both.
