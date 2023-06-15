import math
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


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


def rotate_trajectory(traj, yaw):
    """Rotate a trajectory by yaw."""
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    rotated_traj = np.dot(rotation_matrix, traj.T).T
    return rotated_traj


class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, N, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "processed_images"
        self.N = N
        self.trajectory_data = np.loadtxt(self.root_dir / "trajectories.txt")
        self.valid_indices = self.get_valid_indices()
        self.transform = transform

    def __len__(self):
        return len(self.valid_indices)

    def get_valid_indices(self):
        valid_indices = []

        start, end = 0, 0
        for i, end_of_sequence in enumerate(self.trajectory_data[:, -1]):
            if end_of_sequence:
                end = i
                if end - start >= self.N:  # if we have at least one valid subsequence
                    # Add the start of each valid subsequence of N steps
                    for subseq_start in range(
                        start, end - self.N + 1
                    ):  # +1 bc range end is exclusive
                        valid_indices.append(subseq_start)
                start = end + 1  # set the start of the next sequence

        return valid_indices

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        # Load image
        img_idx = real_idx + 1  # shifted by one
        # order_of_magnitude = math.floor(
        #     math.log10(len(self.trajectory_data))
        # )  # how many zeros in image names
        img_pth = self.image_dir / f"{str(img_idx).zfill(4)}.jpg"
        image = read_image(str(img_pth))
        image = F.convert_image_dtype(image, torch.float)
        if self.transform:
            image = self.transform(image)

        # Load and process trajectory
        trajectory_data = self.trajectory_data[
            real_idx : real_idx + self.N
        ]  # fetch N future steps

        # Extract car positions (x, y) and yaw
        car_positions = trajectory_data[:, :2]
        yaw = trajectory_data[0, 2]  # fetch yaw at the current step

        # Rotate the future car positions into the car's frame
        relative_positions = car_positions - car_positions[0]  # make relative to current position
        rotated_positions = rotate_trajectory(relative_positions, yaw)

        # get the steering
        steering = trajectory_data[:, 4].reshape(-1, 1)
        throttle = trajectory_data[:, 5].reshape(-1, 1)
        trajectory = np.column_stack((rotated_positions, steering, throttle))

        MAX_SPEED = 8
        MIN_SPEED = 2.5
        rescaled_speed = 2 * ((trajectory_data[0, 3] - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) - 1
        rescaled_speed = np.atleast_1d(rescaled_speed)

        # TODO return steering and throttle as different keys
        # do the hstack in the neural net instead?
        # allows using this same code for steering only or steering + throttle
        return {
            "image": image,
            "trajectory": trajectory.flatten().astype(np.float32),  # (N*2)
            "speed": rescaled_speed.astype(np.float32),
        }


class TestTrajectoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.paths = [p for p in self.root_dir.iterdir() if str(p).endswith(".png")]
        self.paths = sorted(self.paths, key=lambda p: int(str(p.name).replace(".png", "")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_pth = self.paths[idx]
        image = read_image(str(img_pth))
        image = F.convert_image_dtype(image, torch.float)
        if self.transform:
            image = self.transform(image)

        return {"image": image}
