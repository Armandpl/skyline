import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_prediction.camera import (
    CENTER_X,
    CENTER_Y,
    CROP_H,
    CROP_LEFT,
    CROP_TOP,
    CROP_W,
    SIM_H,
    SIM_W,
    D,
)

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
view_frame_from_device_frame = device_frame_from_view_frame.T


# plot trajectory on the images

# right, down, forward in meter from center of mass. x is forward
cam_offset = (0, 0.164, -0.105)  # TODO actually offset of the points since we use opencv tvec
cam_rotation = (10.4, 0, 0)  # pitch, yaw, roll in deg
FOCAL_MM = 0.87  # mm
SENSOR_WIDTH = 3.691  # mm
SENSOR_HEIGHT = 2.813
# F(pixels) = F(mm) * ImageWidth (pixel) / SensorWidth(mm)


def plot_traj(image, traj, color=(0, 0, 255)):
    """traj (n ,2) x, y coordinates relative to car center of mass image is a cv2 image so WHC."""
    traj = np.copy(traj)

    traj[:, 1] = -traj[:, 1]  # TODO fix this, or at least understand why its flipped?

    F_x = FOCAL_MM * SIM_W / SENSOR_WIDTH
    F_y = FOCAL_MM * SIM_H / SENSOR_HEIGHT

    traj_3d = np.pad(traj, (0, 1))  # add z = 0
    traj_3d = np.einsum("jk,ik->ij", view_frame_from_device_frame, traj_3d)

    # Compute the camera matrix (assuming square pixels and no skew)
    camera_matrix = np.array(
        [[F_x, 0, SIM_W * CENTER_X], [0, F_y, SIM_H * CENTER_Y], [0, 0, 1]]
    )  # use F computed from sensor size as opposed to from calibration to have the right scale

    # Project the points onto the (uncropped) image plane
    r = R.from_euler("xyz", cam_rotation, degrees=True)
    rvec = r.as_rotvec().astype(np.float32)
    tvec = np.array(cam_offset).reshape(1, 3).astype(np.float32)
    traj_3d = traj_3d.reshape(-1, 1, 3).astype(np.float32)  # (N, 1, 3)
    proj_points, _ = cv2.fisheye.projectPoints(
        objectPoints=traj_3d, rvec=rvec, tvec=tvec, K=camera_matrix, D=D
    )
    proj_points = proj_points.reshape(-1, 2)

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
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius=2, color=color, thickness=-1)


def make_vid_from_2d_traj(trajectory_array_1, trajectory_array_2=None):
    # Initialize a list to store images
    images = []

    # Set up the figure and the axis
    fig, ax = plt.subplots()
    plt.axis("off")

    for t in range(trajectory_array_1.shape[0]):
        ax.clear()
        ax.plot(trajectory_array_1[t, :, 1], trajectory_array_1[t, :, 0], "bo")

        if trajectory_array_2 is not None:
            ax.plot(trajectory_array_2[t, :, 1], trajectory_array_2[t, :, 0], "go")

        min_x = min(
            np.min(trajectory_array_1[:, :, 1]),
            np.min(trajectory_array_2[:, :, 1] if trajectory_array_2 is not None else np.inf),
        )
        max_x = max(
            np.max(trajectory_array_1[:, :, 1]),
            np.max(trajectory_array_2[:, :, 1] if trajectory_array_2 is not None else -np.inf),
        )

        min_y = min(
            np.min(trajectory_array_1[:, :, 0]),
            np.min(trajectory_array_2[:, :, 0] if trajectory_array_2 is not None else np.inf),
        )
        max_y = max(
            np.max(trajectory_array_1[:, :, 0]),
            np.max(trajectory_array_2[:, :, 0] if trajectory_array_2 is not None else -np.inf),
        )

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(data)

    return np.array(images)
