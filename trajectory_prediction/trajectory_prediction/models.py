import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class TransferNet(nn.Module):
    def __init__(self, out_size: int = 1, freeze_base: bool = True):
        super().__init__()
        # Load the pre-trained ResNet18 model
        self.weights = models.ResNet18_Weights.DEFAULT
        self.resnet18 = models.resnet18(weights=self.weights)

        if freeze_base:
            for param in self.resnet18.parameters():
                param.requires_grad = False

        # Replace the last fully connected layer to match the number of classes
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, out_size)

    def forward(self, x):
        # Pass the input through the ResNet18 layers
        x = self.resnet18(x)
        return x

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
