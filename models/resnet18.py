import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Load base ResNet18
        self.model = resnet18(weights=None)

        # Modify the first conv layer for CIFAR (3x3 kernel, stride 1)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove the maxpool layer (CIFAR images are small)
        self.model.maxpool = nn.Identity()

        # Change classifier head
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
