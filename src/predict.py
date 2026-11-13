"""Predict the 2-year and 5-year breast cancer risk with a trained baseline model.

The script reads every exam in a folder of images preprocessed with
preprocessing/preprocess_images_v2.py and writes one row per exam to a CSV
file.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from dataset import load_image
from model import RiskModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Trained model, for example best_model.pt"
    )
    parser.add_argument(
        "--image-root", type=Path, required=True, help="Folder of preprocessed images"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="CSV file for the predictions"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskModel(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    exam_dirs = sorted(path.parent for path in args.image_root.glob("*/*/T1_pre.nii.gz"))
    print(f"Found {len(exam_dirs)} exams.", flush=True)

    rows = []
    for index, exam_dir in enumerate(exam_dirs, start=1):
        image = load_image(exam_dir / "T1_pre.nii.gz", exam_dir / "T1_post_1.nii.gz")
        with torch.no_grad():
            risk_2y, risk_5y = torch.sigmoid(model(image[None].to(device)))[0].tolist()

        # Cancer within 2 years is also cancer within 5 years, so the 5-year
        # risk cannot be lower than the 2-year risk.
        risk_5y = max(risk_2y, risk_5y)

        exam_id = exam_dir.relative_to(args.image_root).as_posix()
        rows.append({"exam_id": exam_id, "risk_2y": risk_2y, "risk_5y": risk_5y})
        print(f"[{index}/{len(exam_dirs)}] {exam_id}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
