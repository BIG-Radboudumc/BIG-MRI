"""Predict the 2-year and 5-year breast cancer risk with a trained baseline model.

The script reads every exam in a folder of images preprocessed with
preprocessing/preprocess_images_v2.py and writes one row per exam to a CSV
file. If several models are given, their predicted risks are averaged.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import load_image
from model import MODELS


class ExamImages(Dataset):
    """The subtraction image of every exam folder in a list."""

    def __init__(self, exam_dirs):
        self.exam_dirs = exam_dirs

    def __len__(self):
        return len(self.exam_dirs)

    def __getitem__(self, index):
        exam_dir = self.exam_dirs[index]
        return load_image(exam_dir / "T1_pre.nii.gz", exam_dir / "T1_post_1.nii.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        nargs="+",
        help="One or more trained models, for example best_model.pt",
    )
    parser.add_argument(
        "--architecture",
        choices=list(MODELS),
        default="3d",
        help="Architecture of the trained models: 3d or 2d",
    )
    parser.add_argument(
        "--image-root", type=Path, required=True, help="Folder of preprocessed images"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="CSV file for the predictions"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Exams per forward pass")
    parser.add_argument(
        "--workers", type=int, default=4, help="Processes that read the images"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in args.model:
        model = MODELS[args.architecture](pretrained=False)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())
    print(f"Loaded {len(models)} model(s).", flush=True)

    exam_dirs = sorted(path.parent for path in args.image_root.glob("*/*/T1_pre.nii.gz"))
    print(f"Found {len(exam_dirs)} exams.", flush=True)

    # The images are read by several processes and predicted in batches. The
    # loader keeps the order of exam_dirs.
    loader = DataLoader(
        ExamImages(exam_dirs), batch_size=args.batch_size, num_workers=args.workers
    )

    rows = []
    for images in loader:
        images = images.to(device)
        with torch.no_grad():
            risks = torch.stack([torch.sigmoid(model(images)) for model in models])

        # Average the predicted risks of all models.
        for risk_2y, risk_5y in risks.mean(dim=0).tolist():
            # Cancer within 2 years is also cancer within 5 years, so the 5-year
            # risk cannot be lower than the 2-year risk.
            risk_5y = max(risk_2y, risk_5y)

            exam_id = exam_dirs[len(rows)].relative_to(args.image_root).as_posix()
            rows.append({"exam_id": exam_id, "risk_2y": risk_2y, "risk_5y": risk_5y})
            print(f"[{len(rows)}/{len(exam_dirs)}] {exam_id}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
