"""Data loading for the baseline model.

The input of the model is the subtraction image (first post-contrast minus
pre-contrast), computed from the output of
preprocessing/preprocess_images_v2.py.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrast,
    RandAffine,
    RandCoarseDropout,
    RandFlip,
    RandGaussianNoise,
    RandZoom,
)
from torch.nn.functional import avg_pool3d
from torch.utils.data import Dataset


DOWNSAMPLE = 2  # 1 mm, 352 x 352 x 176 voxels -> 2 mm, 176 x 176 x 88 voxels


def load_image(pre_path, post_path):
    """Return the subtraction image as a tensor of shape (1, z, y, x)."""
    pre = sitk.GetArrayFromImage(sitk.ReadImage(str(pre_path))).astype(np.float32)
    post = sitk.GetArrayFromImage(sitk.ReadImage(str(post_path))).astype(np.float32)

    # Keep the enhancement only.
    image = np.clip(post - pre, 0, None)

    # Scale intensities to [0, 1] using the 99.5th percentile.
    upper = np.percentile(image, 99.5)
    if upper > 0:
        image = np.clip(image / upper, 0, 1)

    image = torch.from_numpy(image)[None]
    return avg_pool3d(image[None], DOWNSAMPLE)[0]


class RiskDataset(Dataset):
    """Exams listed in a label table, with their 2-year and 5-year labels.

    The table needs the columns exam_id, label_2y and label_5y. The images of
    an exam are read from <image_root>/<exam_id>/.
    """

    def __init__(self, table, image_root, train=False):
        self.table = table.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.augment = None
        if train:
            self.augment = Compose(
                [
                    # Left-right flip.
                    RandFlip(prob=0.5, spatial_axis=2),
                    # Zoom in or out by up to 20%.
                    RandZoom(
                        prob=0.8,
                        min_zoom=0.8,
                        max_zoom=1.2,
                        mode="trilinear",
                        padding_mode="minimum",
                    ),
                    # Shift by up to 4 voxels (8 mm).
                    RandAffine(prob=0.5, translate_range=4, padding_mode="zeros"),
                    # Set a few small blocks to zero.
                    RandCoarseDropout(
                        prob=0.5,
                        holes=4,
                        max_holes=10,
                        spatial_size=(6, 8, 8),
                        fill_value=0,
                    ),
                    RandGaussianNoise(prob=0.5, mean=0.0, std=0.05),
                    RandAdjustContrast(prob=0.5, gamma=(0.7, 1.5)),
                ]
            )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        exam_dir = self.image_root / row["exam_id"]
        image = load_image(exam_dir / "T1_pre.nii.gz", exam_dir / "T1_post_1.nii.gz")
        if self.augment is not None:
            image = self.augment(image).as_tensor()
        labels = torch.tensor([row["label_2y"], row["label_5y"]], dtype=torch.float32)
        return image, labels
