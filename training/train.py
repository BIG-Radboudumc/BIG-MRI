"""Train the baseline model for 2-year and 5-year breast cancer risk prediction.

The model is a 3D ResNet-18 with two outputs. Each output is trained as a
binary classification task: cancer within 2 years, and cancer within 5 years.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from dataset import RiskDataset
from model import RiskModel


LABELS = ["label_2y", "label_5y"]


def split_by_patient(table, fold, seed):
    """Split the table into training and validation exams, by patient."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(table, table["label_5y"], groups=table["patient_id"]))
    train_index, valid_index = splits[fold]
    return table.iloc[train_index], table.iloc[valid_index]


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """Run one epoch. Return the mean loss, the predicted risks and the labels."""
    training = optimizer is not None
    model.train(training)
    losses, risks, targets = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        risks.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(labels.cpu().numpy())

    return np.mean(losses), np.concatenate(risks), np.concatenate(targets)


def compute_auc(targets, risks):
    """Return the AUC, or NaN if only one class is present."""
    if len(np.unique(targets)) < 2:
        return float("nan")
    return roc_auc_score(targets, risks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Label table (CSV)")
    parser.add_argument(
        "--image-root", type=Path, required=True, help="Output folder of preprocessing/v2"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Folder for the model and the log"
    )
    parser.add_argument("--fold", type=int, default=0, help="Validation fold, 0 to 4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Train from random weights"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data: split by patient, so no patient is in both sets.
    table = pd.read_csv(args.labels)
    train_table, valid_table = split_by_patient(table, args.fold, args.seed)
    print(f"Training exams: {len(train_table)}, validation exams: {len(valid_table)}")
    train_loader = DataLoader(
        RiskDataset(train_table, args.image_root, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        RiskDataset(valid_table, args.image_root),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # 2. Model, loss and optimizer. Positive exams are rare, so they get a
    #    larger weight in the loss: the number of negatives per positive.
    model = RiskModel(pretrained=not args.no_pretrained).to(device)
    positives = train_table[LABELS].sum().to_numpy()
    negatives = len(train_table) - positives
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 3. Training loop: keep the model with the best mean validation AUC.
    best_auc = 0.0
    log = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_function, device, optimizer)
        valid_loss, risks, targets = run_epoch(model, valid_loader, loss_function, device)
        scheduler.step()

        auc_2y = compute_auc(targets[:, 0], risks[:, 0])
        auc_5y = compute_auc(targets[:, 1], risks[:, 1])
        mean_auc = np.nanmean([auc_2y, auc_5y])
        print(
            f"Epoch {epoch}/{args.epochs}: train loss {train_loss:.4f}, "
            f"validation loss {valid_loss:.4f}, "
            f"AUC 2y {auc_2y:.3f}, AUC 5y {auc_5y:.3f}",
            flush=True,
        )

        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "auc_2y": auc_2y,
                "auc_5y": auc_5y,
            }
        )
        pd.DataFrame(log).to_csv(args.output_dir / "log.csv", index=False)

        torch.save(model.state_dict(), args.output_dir / "last_model.pt")
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    print(f"Finished. Best mean validation AUC: {best_auc:.3f}")


if __name__ == "__main__":
    main()
