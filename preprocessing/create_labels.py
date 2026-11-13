"""Create the label table of the baseline from the BIG-MRI clinical table.

The clinical table has one row per lesion. This script writes one row per
exam, with the 2-year and 5-year labels, the time to cancer, the follow-up
time, and the data split.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


DAYS_PER_YEAR = 365.25
FOLLOWUP_AFTER_LAST_EXAM = 1.0  # Years of follow-up assumed after the last exam
TEST_FRACTION = 0.2  # Fraction of the patients in the independent test set
FOLDS = 4  # Cross-validation folds for the remaining patients


def read_exams(clinical_path):
    """Read the clinical table and return one row per exam."""
    clinical = pd.read_excel(clinical_path)
    clinical["exam_date"] = pd.to_datetime(clinical["exam_date"], format="%Y-%m-%d")
    # A missing diagnosis date is written as "not provided" and becomes empty.
    clinical["cancer_date"] = pd.to_datetime(
        clinical["Cancer Detection Date"], format="%Y-%m-%d", errors="coerce"
    )

    exams = (
        clinical.groupby("exam_folder")
        .agg(
            patient_id=("release_id", "first"),
            exam_date=("exam_date", "first"),
            cancer_present=("cancer_present", "max"),
        )
        .reset_index()
        .rename(columns={"exam_folder": "exam_id"})
    )

    # Each patient has at most one diagnosis date.
    cancer_date = clinical.groupby("release_id")["cancer_date"].min()
    exams["cancer_date"] = exams["patient_id"].map(cancer_date)
    return exams


def add_labels(exams):
    """Add the time to cancer, the follow-up time and the two labels."""
    days_to_diagnosis = (exams["cancer_date"] - exams["exam_date"]).dt.days

    # Time from the exam to cancer, in years. Empty if there is no cancer.
    #   - Cancer is present on the exam: 0.
    #   - Cancer is diagnosed after the exam: time to the diagnosis.
    #   - Otherwise (no diagnosis, or a diagnosis before the exam): no cancer.
    years_to_cancer = (days_to_diagnosis / DAYS_PER_YEAR).where(days_to_diagnosis > 0)
    years_to_cancer = years_to_cancer.mask(exams["cancer_present"] == 1, 0.0)

    # An exam without cancer has an empty time, which compares as False.
    exams["label_2y"] = (years_to_cancer < 2).astype(int)
    exams["label_5y"] = (years_to_cancer < 5).astype(int)
    exams["years_to_cancer"] = years_to_cancer.round(3)

    # Follow-up: until one year after the last exam of the patient.
    last_exam = exams.groupby("patient_id")["exam_date"].transform("max")
    days_to_last_exam = (last_exam - exams["exam_date"]).dt.days
    exams["years_to_last_followup"] = (
        days_to_last_exam / DAYS_PER_YEAR + FOLLOWUP_AFTER_LAST_EXAM
    ).round(3)
    return exams


def add_split(exams, seed):
    """Assign every patient to the test set or to a cross-validation fold."""
    # Split by patient, with a similar share of cancer patients in every part.
    patients = exams.groupby("patient_id")["label_5y"].max().reset_index()
    development, test = train_test_split(
        patients,
        test_size=TEST_FRACTION,
        stratify=patients["label_5y"],
        random_state=seed,
    )

    fold = {patient: "test" for patient in test["patient_id"]}
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    development = development.reset_index(drop=True)
    for number, (_, index) in enumerate(
        splitter.split(development, development["label_5y"])
    ):
        for patient in development.loc[index, "patient_id"]:
            fold[patient] = str(number)

    exams["fold"] = exams["patient_id"].map(fold)
    return exams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clinical", type=Path, required=True, help="Clinical table (Excel)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="CSV file for the label table"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the data split")
    args = parser.parse_args()

    exams = add_split(add_labels(read_exams(args.clinical)), args.seed)
    columns = [
        "patient_id",
        "exam_id",
        "label_2y",
        "label_5y",
        "years_to_cancer",
        "years_to_last_followup",
        "fold",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exams[columns].to_csv(args.output, index=False)

    print(f"Wrote {len(exams)} exams of {exams['patient_id'].nunique()} patients.")
    summary = exams.groupby("fold").agg(
        patients=("patient_id", "nunique"),
        exams=("exam_id", "size"),
        positive_2y=("label_2y", "sum"),
        positive_5y=("label_5y", "sum"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
