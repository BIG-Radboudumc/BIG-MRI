# Evaluation

Evaluation and ranking scripts for BIG-MRI.

[`evaluate.py`](evaluate.py) compares a prediction table with a label table and reports the AUC for the 2-year and the 5-year risk.

## Input

1. A prediction table written by [`inference/predict.py`](../inference/), with the columns `exam_id`, `risk_2y`, and `risk_5y`.
2. A label table in CSV format, with the columns `exam_id`, `label_2y`, and `label_5y`. This is the same table that is used for training.

Only exams that appear in both tables are evaluated. Evaluate on exams that were not used to train the model, for example by passing a label table that contains only the validation exams.

## Usage

```bash
python evaluation/evaluate.py --predictions /path/to/predictions.csv --labels /path/to/labels.csv
```

Add `--output /path/to/metrics.json` to also save the metrics to a file.

## Metrics

| Metric | Definition |
| --- | --- |
| `auc_2y` | Area under the ROC curve of `risk_2y` against `label_2y`. |
| `auc_5y` | Area under the ROC curve of `risk_5y` against `label_5y`. |

Harrell's C-index and the leaderboard computation details are coming soon.

## How the AUC is computed

The AUC is computed separately for each horizon (2 years and 5 years). The explanation below uses the 5-year horizon; the 2-year AUC works in the same way.

### Definition

Split the exams into two groups by their label:

- **Positive exams**: breast cancer within 5 years (`label_5y = 1`).
- **Negative exams**: no breast cancer within 5 years (`label_5y = 0`).

Form every pair that consists of one positive exam and one negative exam. For each pair, compare the two predicted risks:

- the pair counts 1 if the positive exam has the higher risk,
- the pair counts 0.5 if both exams have the same risk,
- the pair counts 0 if the positive exam has the lower risk.

The AUC is the sum of these counts divided by the number of pairs:

```text
AUC = (pairs ranked correctly + 0.5 * tied pairs) / (positive exams * negative exams)
```

This number is equal to the area under the ROC curve, which plots the sensitivity against 1 minus the specificity for every possible risk threshold (Hanley and McNeil, 1982).

### Interpretation

The AUC is the probability that the model assigns a higher risk to a randomly chosen exam that is followed by cancer within 5 years than to a randomly chosen exam that is not.

- 1.0: every positive exam has a higher risk than every negative exam.
- 0.5: the ranking is no better than chance.
- Below 0.5: the ranking is worse than chance.

The AUC depends only on the order of the predicted risks. It does not change when all risks are rescaled, so it measures discrimination and not calibration.

### Example

Six exams, three positive and three negative:

| Exam | Label | Risk |
| --- | ---: | ---: |
| A | 0 | 0.10 |
| B | 0 | 0.20 |
| C | 0 | 0.75 |
| D | 1 | 0.70 |
| E | 1 | 0.75 |
| F | 1 | 0.90 |

There are 3 x 3 = 9 pairs. Seven are ranked correctly. The pair (D, C) is ranked incorrectly, because the negative exam C has a higher risk than the positive exam D. The pair (E, C) is tied.

```text
AUC = (7 + 0.5 * 1) / 9 = 0.833
```

### Relation to the time-dependent AUC

Risk models are usually evaluated with an AUC per time horizon. The reference implementations of Mirai and Sybil define it as follows for a horizon of *t* years:

- an exam is positive if cancer is diagnosed within *t* years,
- an exam is negative if it is known to be cancer-free at *t* years, either because the follow-up is at least *t* years or because cancer is diagnosed later than *t* years,
- an exam without cancer and with less than *t* years of follow-up is excluded, because its status at *t* years is unknown.

The AUC is then the ROC AUC of the predicted risk at *t* years on the remaining exams.

`evaluate.py` computes the same number when the status of every exam at the horizon is known, which is what the binary labels `label_2y` and `label_5y` express. It does not use follow-up times, so it cannot exclude exams with insufficient follow-up: an exam with `label_5y = 0` is always treated as cancer-free at 5 years.

### Implementation

The AUC is computed with [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), the same function that the Mirai and Sybil reference implementations call after selecting the exams.

## References

- Yala A, et al. Towards Robust Mammography-Based Models for Breast Cancer Risk. Science Translational Medicine, 2021. [doi:10.1126/scitranslmed.aba4373](https://www.science.org/doi/10.1126/scitranslmed.aba4373). Code: [yala/Mirai](https://github.com/yala/Mirai), function `compute_auc_x_year_auc` in `onconet/learn/utils.py`.
- Mikhael PG, et al. Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography. Journal of Clinical Oncology, 2023. Code: [reginabarzilaygroup/Sybil](https://github.com/reginabarzilaygroup/Sybil), function `compute_auc_at_followup` in `sybil/utils/metrics.py`.
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 1982;143(1):29-36.
