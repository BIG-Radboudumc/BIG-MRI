# Evaluation metrics

This document explains the metrics reported by [`src/evaluate.py`](../src/evaluate.py).

| Metric | Meaning |
| --- | --- |
| `auc_2y` | AUC of the predicted 2-year risk. |
| `auc_5y` | AUC of the predicted 5-year risk. |
| `c_index` | Harrell's C-index of the predicted 5-year risk over the full follow-up. |

Both metrics use two columns of the label table: `years_to_cancer` (empty if there is no cancer) and `years_to_last_followup`. In the label table of the baseline, the follow-up is assumed to end one year after the last exam of the patient, so both metrics depend on that assumption.

## How the AUC is computed

The AUC is computed separately for each horizon (2 years and 5 years). The explanation below uses the 5-year horizon; the 2-year AUC works in the same way.

### Definition

Split the exams into three groups:

- **Positive exams**: breast cancer within 5 years.
- **Negative exams**: known to be cancer-free at 5 years, either because the follow-up is at least 5 years or because cancer is diagnosed later than 5 years.
- **Excluded exams**: no cancer and less than 5 years of follow-up. Their status at 5 years is unknown.

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

The AUC is the probability that the model assigns a higher risk to a randomly chosen exam that is followed by cancer within 5 years than to a randomly chosen exam that is known to be cancer-free at 5 years.

- 1.0: every positive exam has a higher risk than every negative exam.
- 0.5: the ranking is no better than chance.
- Below 0.5: the ranking is worse than chance.

The AUC depends only on the order of the predicted risks. It does not change when all risks are rescaled, so it measures discrimination and not calibration.

### Example

Seven exams, evaluated at 5 years:

| Exam | Cancer | Follow-up | Group | Risk |
| --- | --- | --- | --- | ---: |
| A | no | 6 years | negative | 0.10 |
| B | no | 8 years | negative | 0.20 |
| C | after 7 years | | negative | 0.75 |
| D | after 1 year | | positive | 0.70 |
| E | after 3 years | | positive | 0.75 |
| F | after 4 years | | positive | 0.90 |
| G | no | 3 years | excluded | 0.95 |

Exam G is excluded. There are 3 x 3 = 9 pairs of a positive and a negative exam. Seven are ranked correctly. The pair (D, C) is ranked incorrectly, because the negative exam C has a higher risk than the positive exam D. The pair (E, C) is tied.

```text
AUC = (7 + 0.5 * 1) / 9 = 0.833
```

### Relation to the reference implementations

This is the AUC per time horizon of the Mirai and Sybil reference implementations (functions `compute_auc_x_year_auc` and `compute_auc_at_followup`). They apply the same three groups and then compute the ROC AUC of the predicted risk at that horizon on the positive and negative exams.

When every exam without cancer has enough follow-up, no exam is excluded and the result is the ordinary ROC AUC of the predicted risk against the binary label.

### Implementation

After selecting the positive and negative exams, the AUC is computed with [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), the same function that the Mirai and Sybil reference implementations call.

## How the C-index is computed

The AUC looks at one horizon at a time. Harrell's C-index summarizes the ranking over the whole follow-up with one number. It uses one risk score per exam, for which `evaluate.py` takes the predicted 5-year risk.

### Definition

Every exam has an observed time:

- the time to cancer, if cancer is diagnosed,
- the follow-up time, if no cancer is diagnosed. Such an exam is called censored: it is only known that the exam was cancer-free up to that time.

Form every pair of two exams. A pair is **comparable** if it is known which of the two exams had cancer first. This is the case when the exam with the shorter observed time had cancer. A pair is not comparable when the exam with the shorter observed time is censored, or when both exams have the same observed time.

For each comparable pair, compare the two predicted risks:

- the pair counts 1 if the exam with cancer first has the higher risk,
- the pair counts 0.5 if both exams have the same risk,
- the pair counts 0 if the exam with cancer first has the lower risk.

```text
C-index = (pairs ranked correctly + 0.5 * tied pairs) / (comparable pairs)
```

### Interpretation

The C-index is the probability that, of two randomly chosen comparable exams, the model assigns the higher risk to the exam that is followed by cancer first.

- 1.0: the risks are in the same order as the times to cancer.
- 0.5: the ranking is no better than chance.

Like the AUC, the C-index depends only on the order of the predicted risks. Unlike the AUC, it also rewards ranking an exam with cancer after 1 year above an exam with cancer after 4 years.

### Example

Five exams:

| Exam | Cancer | Follow-up | Observed time | Risk |
| --- | --- | --- | --- | ---: |
| A | after 1 year | | 1 year | 0.9 |
| B | after 3 years | | 3 years | 0.4 |
| C | no | 2 years | 2 years, censored | 0.5 |
| D | no | 6 years | 6 years, censored | 0.2 |
| E | after 4 years | | 4 years | 0.6 |

Of the 10 pairs, 7 are comparable: (A, B), (A, C), (A, D), (A, E), (B, D), (B, E), and (E, D). The pairs (B, C), (C, D), and (C, E) are not comparable, because exam C is censored at 2 years and has the shorter observed time in each of them.

Six comparable pairs are ranked correctly. The pair (B, E) is ranked incorrectly: exam B has cancer first, but exam E has the higher risk.

```text
C-index = 6 / 7 = 0.857
```

### Implementation

The C-index is computed with [`lifelines.utils.concordance_index`](https://lifelines.readthedocs.io/en/latest/lifelines.utils.html#lifelines.utils.concordance_index), which implements Harrell's C-index (Harrell et al., 1996).

The Mirai and Sybil reference implementations report a different variant under the same name: it weights the pairs by the inverse probability of censoring, uses the predicted risk of each year, and is restricted to their follow-up window. Its values are therefore not directly comparable to Harrell's C-index.

## References

- Yala A, et al. Towards Robust Mammography-Based Models for Breast Cancer Risk. Science Translational Medicine, 2021. [doi:10.1126/scitranslmed.aba4373](https://www.science.org/doi/10.1126/scitranslmed.aba4373). Code: [yala/Mirai](https://github.com/yala/Mirai), function `compute_auc_x_year_auc` in `onconet/learn/utils.py`.
- Mikhael PG, et al. Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography. Journal of Clinical Oncology, 2023. Code: [reginabarzilaygroup/Sybil](https://github.com/reginabarzilaygroup/Sybil), function `compute_auc_at_followup` in `sybil/utils/metrics.py`.
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 1982;143(1):29-36.
- Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Statistics in Medicine, 1996;15(4):361-387.
- Davidson-Pilon C. lifelines: survival analysis in Python. Code: [CamDavidsonPilon/lifelines](https://github.com/CamDavidsonPilon/lifelines).
