import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class TransferNet(nn.Module):
    def __init__(self, out_size: int = 1, out_scale: float = 1, freeze_base: bool = True):
        super().__init__()
        self.out_scale = out_scale
        # Load the pre-trained ResNet18 model
        weights = models.ResNet18_Weights.DEFAULT
        resnet18 = models.resnet18(weights=weights)

        if freeze_base:
            for param in resnet18.parameters():
                param.requires_grad = False

        # replace the last fully connected layer
        num_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_features, out_size)
        self.model = nn.Sequential(resnet18, nn.Tanh())

    def forward(self, x):
        x = self.model(x) * self.out_scale  # allows predicting up to out_scale meters aways
        # might be smarter to scale the labels instead idk
        return x

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # imagenet
            ]
        )


class LongiNet(nn.Module):
    def __init__(self, out_size: int = 1, out_scale: float = 1, freeze_base: bool = True):
        super().__init__()
        self.out_scale = out_scale
        # Load the pre-trained ResNet18 model
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet18 = models.resnet18(weights=weights)

        if freeze_base:
            for param in self.resnet18.parameters():
                param.requires_grad = False

        # replace the last fully connected layer
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.fc = nn.Linear(num_features + 1, out_size)
        # resnet18.fc = nn.Linear(num_features, out_size)
        # self.model = nn.Sequential(resnet18, nn.Tanh())

    def forward(self, image, speed):
        features = self.resnet18(image)
        out = self.fc(torch.concat((features, speed), axis=1))
        return torch.tanh(out) * self.out_scale

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # imagenet
            ]
        )
