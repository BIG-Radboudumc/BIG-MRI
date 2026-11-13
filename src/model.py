"""Baseline model: a 3D ResNet-18 that predicts 2-year and 5-year risk."""

from monai.networks.nets import resnet18
from torch import nn


class RiskModel(nn.Module):
    """3D ResNet-18 with one output per prediction horizon."""

    def __init__(self, pretrained=True):
        super().__init__()
        # These settings match the public MedicalNet ResNet-18 weights, which
        # are downloaded automatically when pretrained is True.
        self.encoder = resnet18(
            pretrained=pretrained,
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=2,  # Downsample equally in all three directions.
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True,
        )
        self.head = nn.Linear(512, 2)

    def forward(self, image):
        """Return logits of shape (batch, 2): 2-year risk and 5-year risk."""
        return self.head(self.encoder(image))
