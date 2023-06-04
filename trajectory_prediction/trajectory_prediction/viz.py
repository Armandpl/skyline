import cv2
import numpy as np
import torch
from kornia.geometry.camera.perspective import project_points
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image
from tqdm import trange

from trajectory_prediction.camera import (
    CENTER_X,
    CENTER_Y,
    CROP_H,
    CROP_LEFT,
    CROP_TOP,
    CROP_W,
    SIM_H,
    SIM_W,
)

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
view_frame_from_device_frame = device_frame_from_view_frame.T


def transform_points(points, offset, rotation):
    """Transform the points from one frame of reference to another.

    Args:
        points (Tensor): the points to be transformed, with shape (n, 3).
        offset (tuple): the translation vector.
        rotation (tuple): the rotation angles in degrees.

    Returns:
        Tensor: the transformed points, with the same shape as the input points.
    """
    # Create the transformation matrix
    offset = torch.tensor(offset).float()
    rotation = R.from_euler("xyz", rotation, degrees=True).as_matrix()
    transform = torch.from_numpy(rotation).float()

    # Apply the transformation
    transformed_points = torch.mm(points - offset, transform.T)

    return transformed_points


# plot trajectory on the images
cam_offset = (0, -0.164, 0.105)  # right, down, forward in meter from center of mass. x is forward
cam_rotation = (10.4, 0, 0)  # pitch, yaw, roll in deg
FOCAL_MM = 0.87  # mm
SENSOR_WIDTH = 3.691  # mm
SENSOR_HEIGHT = 2.813
# F(pixels) = F(mm) * ImageWidth (pixel) / SensorWidth(mm)


def plot_traj(image, traj, color=(0, 0, 255)):
    """traj (n ,2) x, y coordinates relative to car center of mass image is a cv2 image so WHC."""

    traj[:, 1] = -traj[:, 1]  # TODO fix this, or at least understand why its flipped?

    F_x = FOCAL_MM * SIM_W / SENSOR_WIDTH
    F_y = FOCAL_MM * SIM_H / SENSOR_HEIGHT

    traj_3d = np.pad(traj, (0, 1))  # add z = 0
    traj_3d = np.einsum("jk,ik->ij", view_frame_from_device_frame, traj_3d)
    traj_3d = torch.from_numpy(traj_3d).float()  # to use kornia

    # Transform the points to the camera's frame
    traj_cam = transform_points(traj_3d, cam_offset, cam_rotation)

    # Compute the camera matrix (assuming square pixels and no skew)
    camera_matrix = torch.tensor(
        [[F_x, 0, SIM_W * CENTER_X], [0, F_y, SIM_H * CENTER_Y], [0, 0, 1]]
    )

    # Project the points onto the (uncropped) image plane
    proj_points = project_points(traj_cam, camera_matrix)

    # Distort points w/ D
    # TODO

    # crop the projected points
    proj_points[:, 0] -= CROP_LEFT
    proj_points[:, 1] -= CROP_TOP

    # resize the projected points to fit the current image size
    w, h, _ = image.shape
    resize_scale_x = w / CROP_W
    resize_scale_y = h / CROP_H
    proj_points[:, 0] *= resize_scale_x
    proj_points[:, 1] *= resize_scale_y

    # only keep points inside the frame
    valid_x = np.logical_and(proj_points[:, 0] >= 0, proj_points[:, 0] <= w)
    valid_y = np.logical_and(proj_points[:, 1] >= 0, proj_points[:, 1] <= h)
    valid_points = np.logical_and(valid_x, valid_y)
    proj_points = proj_points[valid_points]

    # plot points on the image
    for point in proj_points:
        x, y = int(point[0].item()), int(point[1].item())
        cv2.circle(image, (x, y), radius=2, color=color, thickness=-1)
