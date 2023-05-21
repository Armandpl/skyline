from pathlib import Path

import numpy as np
import torchvision
from torch.utils.data import Dataset


class SteeringDataset(Dataset):
    def __init__(
        self,
        root_dir="../data/trajectory_optimization/data/centerline_trajectories",
        labels_file="../data/trajectory_optimization/data/centerline_trajectories.txt",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.labels = np.loadtxt(labels_file).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_idx = str(idx + 1).zfill(4)
        img_path = self.root_dir / f"{img_idx}.png"
        image = torchvision.io.read_image(str(img_path))
        label = np.atleast_1d(self.labels[idx, -1])

        if self.transform:
            image = self.transform(image)

        return image, label
