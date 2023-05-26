import cv2
import numpy as np
import torch
from kornia.geometry.camera.perspective import project_points
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image
from tqdm import trange

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
cam_offset = (0, -0.170, 0.105)  # right, down, forward in meter from center of mass. x is forward
cam_rotation = (12, 0, 0)  # pitch, yaw, roll in deg
FOCAL_MM = 0.8726  # mm
sensor_width = 3.92  # mm
sensor_height = 3.92 * (9 / 16)
# F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth (pixel)
# F(pixels) = F(mm) * ImageWidth (pixel) / SensorWidth(mm)


def plot_traj(image, traj, color=(0, 0, 255)):
    """traj (n ,2) x, y coordinates relative to car center of mass image is a cv2 image so WHC."""

    traj[:, 1] = -traj[:, 1]  # TODO fix this, or at least understand why its flipped?

    W, H, _ = image.shape

    F_x = FOCAL_MM * W / sensor_width
    F_y = FOCAL_MM * H / sensor_height

    traj_3d = np.pad(traj, (0, 1))  # add z = 0
    traj_3d = np.einsum("jk,ik->ij", view_frame_from_device_frame, traj_3d)
    traj_3d = torch.from_numpy(traj_3d).float()  # to use kornia

    # Transform the points to the camera's frame
    traj_cam = transform_points(traj_3d, cam_offset, cam_rotation)

    # Compute the camera matrix (assuming square pixels and no skew)
    camera_matrix = torch.tensor([[F_x, 0, W / 2], [0, F_y, H / 2], [0, 0, 1]])

    # Project the points onto the image plane
    proj_points = project_points(traj_cam, camera_matrix)

    # only keep points inside the frame
    valid_x = np.logical_and(proj_points[:, 0] >= 0, proj_points[:, 0] <= W)
    valid_y = np.logical_and(proj_points[:, 1] >= 0, proj_points[:, 1] <= H)

    valid_points = np.logical_and(valid_x, valid_y)

    proj_points = proj_points[valid_points]

    for point in proj_points:
        x, y = int(point[0].item()), int(point[1].item())
        cv2.circle(image, (x, y), radius=2, color=color, thickness=-1)
