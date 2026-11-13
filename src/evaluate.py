"""Evaluate predicted 2-year and 5-year breast cancer risks.

The script compares a prediction table with a label table and reports the
AUC for each horizon and Harrell's C-index.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


HORIZONS = {"2y": 2, "5y": 5}  # Name and length in years


def compute_auc(table, horizon, years):
    """Return the AUC at one horizon, the exams used, and the positive exams."""
    cancer = table["years_to_cancer"]
    followup = table["years_to_last_followup"]

    # Positive: cancer within the horizon.
    positive = cancer < years
    # Negative: known to be cancer-free at the horizon, because cancer is
    # diagnosed later or because the follow-up is long enough.
    negative = (cancer >= years) | (cancer.isna() & (followup >= years))
    # All other exams are excluded: their status at the horizon is unknown.
    known = positive | negative

    if positive.sum() == 0 or negative.sum() == 0:
        raise SystemExit(f"The AUC for {horizon} needs positive and negative exams.")
    auc = roc_auc_score(positive[known], table.loc[known, f"risk_{horizon}"])
    return auc, int(known.sum()), int(positive.sum())


def compute_c_index(table):
    """Return Harrell's C-index of the 5-year risk and the exams with cancer."""
    cancer = table["years_to_cancer"]
    event = cancer.notna()
    if not event.any():
        raise SystemExit("The C-index needs at least one exam with cancer.")

    # Observed time: time to cancer, or the follow-up time if there is no cancer.
    time = cancer.fillna(table["years_to_last_followup"])
    # lifelines expects a score that is higher when cancer comes later.
    return concordance_index(time, -table["risk_5y"], event), int(event.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Output of predict.py"
    )
    parser.add_argument("--labels", type=Path, required=True, help="Label table (CSV)")
    parser.add_argument(
        "--fold", help="Evaluate only this part of the label table, for example test"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON file for the metrics")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    labels = pd.read_csv(args.labels, dtype={"fold": str})
    if args.fold is not None:
        labels = labels[labels["fold"] == args.fold]

    # Evaluate the exams that have both a prediction and a label.
    table = labels.merge(predictions, on="exam_id")
    print(f"Exams with a label: {len(labels)}, with a prediction: {len(predictions)}")
    print(f"Exams evaluated: {len(table)}")

    metrics = {"exams": len(table)}
    for horizon, years in HORIZONS.items():
        auc, exams, positives = compute_auc(table, horizon, years)
        metrics[f"exams_{horizon}"] = exams
        metrics[f"positives_{horizon}"] = positives
        metrics[f"auc_{horizon}"] = auc
        print(f"AUC {horizon}: {auc:.3f} ({exams} exams, {positives} positive)")

    c_index, events = compute_c_index(table)
    metrics["exams_with_cancer"] = events
    metrics["c_index"] = c_index
    print(f"C-index: {c_index:.3f} ({len(table)} exams, {events} with cancer)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote the metrics to {args.output}")


if __name__ == "__main__":
    main()
