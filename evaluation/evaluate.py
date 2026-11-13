"""Evaluate predicted 2-year and 5-year breast cancer risks.

The script compares a prediction table with a label table and reports the
area under the ROC curve (AUC) for each horizon.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


HORIZONS = ["2y", "5y"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Output of inference/predict.py"
    )
    parser.add_argument("--labels", type=Path, required=True, help="Label table (CSV)")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the metrics")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    labels = pd.read_csv(args.labels)

    # Evaluate the exams that have both a prediction and a label.
    table = labels.merge(predictions, on="exam_id")
    print(f"Exams with a label: {len(labels)}, with a prediction: {len(predictions)}")
    print(f"Exams evaluated: {len(table)}")

    metrics = {"exams": len(table)}
    for horizon in HORIZONS:
        label = table[f"label_{horizon}"]
        risk = table[f"risk_{horizon}"]
        if label.nunique() < 2:
            raise SystemExit(
                f"The AUC for {horizon} needs both positive and negative exams."
            )
        metrics[f"positives_{horizon}"] = int(label.sum())
        metrics[f"auc_{horizon}"] = roc_auc_score(label, risk)
        print(
            f"AUC {horizon}: {metrics[f'auc_{horizon}']:.3f} "
            f"({metrics[f'positives_{horizon}']} positive exams)"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote the metrics to {args.output}")


if __name__ == "__main__":
    main()
