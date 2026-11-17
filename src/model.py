"""Baseline models: a 3D ResNet-18 and a 2D ResNet-18 on slices that predict
2-year and 5-year risk."""

import torchvision
from monai.networks.nets import resnet18
from torch import nn


# Mean and standard deviation of the ImageNet images, per channel.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


class SliceRiskModel(nn.Module):
    """2D ResNet-18 on every axial slice, followed by mean pooling over the slices."""

    def __init__(self, pretrained=True):
        super().__init__()
        # The public ImageNet weights are downloaded automatically when
        # pretrained is True.
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = torchvision.models.resnet18(weights=weights)
        self.encoder.fc = nn.Identity()  # Keep the 512 features of every slice.
        self.head = nn.Linear(512, 2)

    def forward(self, image):
        """Return logits of shape (batch, 2): 2-year risk and 5-year risk."""
        batch, _, slices, height, width = image.shape

        # Treat every axial slice as a 2D image. The ImageNet weights expect
        # three channels, so the slice is repeated three times.
        image = image.transpose(1, 2).reshape(batch * slices, 1, height, width)
        image = image.expand(-1, 3, -1, -1)

        # Scale the slice like the ImageNet images.
        mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        features = self.encoder((image - mean) / std)

        # Average the features of all slices of an exam.
        features = features.reshape(batch, slices, -1).mean(dim=1)
        return self.head(features)


MODELS = {"3d": RiskModel, "2d": SliceRiskModel}
