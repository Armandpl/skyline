import random

import bpy
import mathutils
import numpy as np


def plot_line(filepath, name):
    # Your list of vertices. Note that we've added a z-component of 0 to each vertex.
    inner = np.loadtxt(filepath)
    inner = np.pad(inner, ((0, 0), (0, 1)))

    vertices = inner

    # Create a new curve data object
    curve_data = bpy.data.curves.new("my_curve", type="CURVE")
    curve_data.dimensions = "2D"
    curve_data.resolution_u = 2

    # Create a new spline for the curve
    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(vertices) - 1)

    # Assign the vertices to the spline
    for i, coord in enumerate(vertices):
        x, y, z = coord
        polyline.points[i].co = (x, y, z, 1)

    # Create a new object with the curve data and link it to the scene
    curve_object = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_object)

    # Set the curve's bevel depth to make it 5 cm wide
    curve_object.data.bevel_depth = 0.05 / 2  # 5 cm

    # Create a new white material
    # material = bpy.data.materials.new(name="White Material")
    # material.diffuse_color = (1, 1, 1, 1)  # RGBA white color

    # Assign the material to the curve
    # curve_object.data.materials.append(material)

    # flatten the tube, make it a strip
    curve_object.scale = (1, 1, 0.01)


# plot_line("/Users/armandpl/Dev/skyline/trajectory_optimization/notebooks/inner.txt", "inner")
# plot_line("/Users/armandpl/Dev/skyline/trajectory_optimization/notebooks/outer.txt", "outer")


def create_augmentations(traj_path):
    traj = np.loadtxt(traj_path)  # [x, y, yaw]
    end_of_sequence = traj[:, -1]

    # Make sure you have as many frames as you have positions and orientations
    bpy.context.scene.frame_end = traj.shape[0]

    # select relevant objects here
    ground_1 = bpy.data.objects["ground_1"]
    ground_2 = bpy.data.objects["ground_2"]
    ground_2_material = bpy.data.materials["ground_2_material"]
    sun_light = bpy.data.lights["Sun"]
    line_material = bpy.data.materials["line_material"]

    # Iterate over each frame
    for frame in range(1, traj.shape[0]):
        if frame % 1000 == 0:
            print(f"{frame} / {traj.shape[0]}")

        # Augmentations!
        if end_of_sequence[frame - 1]:
            # Set the current frame
            bpy.context.scene.frame_set(frame)

            # Ground augmentation
            ground_1.hide_render = True
            ground_2.hide_render = True

            # randomly show either ground_1 or ground_2
            if random.random() < 0.5:
                ground_1.hide_render = False
            else:
                ground_2.hide_render = False
                ground_2_material.keyframe_insert(
                    data_path="diffuse_color", frame=frame - 1
                )  # keyframe
                grey_value = random.uniform(0, 0.5)  # light grey to dark grey
                random_grey = [grey_value, grey_value, grey_value]  # RGB
                ground_2_material.diffuse_color = random_grey + [1.0]  # Add alpha channel
                ground_2_material.keyframe_insert(
                    data_path="diffuse_color", frame=frame
                )  # keyframe

            # Insert keyframes for the objects' visibility
            ground_1.keyframe_insert(data_path="hide_render", frame=frame)
            ground_2.keyframe_insert(data_path="hide_render", frame=frame)

            # Light augmentation
            sun_light.keyframe_insert(data_path="energy", frame=frame - 1)
            sun_light.keyframe_insert(data_path="color", frame=frame - 1)
            color_options = [
                [1, 1, 1],  # white
                [0.8, 0.8, 1],  # slightly blue
                [1, 1, 0.8],
            ]  # slightly yellow

            chosen_color = random.choice(color_options)

            sun_light.color = chosen_color
            sun_light.energy = random.uniform(0, 10)

            # Insert a keyframe for the light's energy and color
            sun_light.keyframe_insert(data_path="energy", frame=frame)
            sun_light.keyframe_insert(data_path="color", frame=frame)

            # Line augmentation

            line_material.keyframe_insert(data_path="diffuse_color", frame=frame - 1)  # keyframe
            if random.random() > 0.5:
                base_color = random.uniform(
                    0.75, 1.25
                )  # This will result in a range of colors around white/beige
            else:
                base_color = 0  # black lines
            line_material.diffuse_color = [
                base_color,
                base_color,
                base_color,
                1.0,
            ]  # Add alpha channel
            line_material.keyframe_insert(data_path="diffuse_color", frame=frame)  # keyframe


def animate_camera(traj_path):
    traj = np.loadtxt(traj_path)  # [x, y, yaw, speed, delta, speed_command, end_of_seq]

    # Assume you have lists of car positions and orientations
    car_positions = traj[:, 0:2]  # List of Vector((x, y)) for each frame
    car_positions = np.pad(car_positions, ((0, 0), (0, 1)))  # add z=0
    car_orientations = traj[:, 2].reshape(-1, 1)  # List of Euler(yaw) for each frame
    # car_orientations = np.radians(car_orientations)
    car_orientations = np.pad(car_orientations, ((0, 0), (2, 0)))  # add pitch = 0, roll = 0
    end_of_sequence = traj[:, -1]

    # Camera offset from car
    camera_offset_position = mathutils.Vector((0.105, 0, 0.164))  # x, y, z
    camera_offset_rotation = mathutils.Euler(
        (np.radians(90) + np.radians(-10.4), np.radians(0), np.radians(-90))
    )

    camera = bpy.data.objects["Camera"]
    # Create two empty lists to store final camera positions and rotations
    camera_positions_world = []
    camera_orientations_world = []

    # Iterate over each frame
    for frame in range(len(car_positions)):
        # Create rotation matrix from car's orientation
        car_rot_matrix = mathutils.Matrix.Rotation(
            car_orientations[frame][2], 4, "Z"
        )  # 4 for 4x4 matrix, 'Z' for rotation about Z-axis

        # Transform camera offset from car's coordinate frame to world coordinate frame
        camera_offset_position_world = car_rot_matrix @ camera_offset_position

        # Compute camera position and rotation based on car's data and transformed offset
        camera_positions_world.append(car_positions[frame] + camera_offset_position_world)
        camera_orientations_world.append(car_orientations[frame] + camera_offset_rotation)

    print("done creating pos/rot")

    # Prepare your data: flatten the list of camera positions and orientations
    frames = range(1, len(camera_positions_world))
    flattened_positions = [
        [item[i] for item in camera_positions_world] for i in range(3)
    ]  # Flatten each component separately
    flattened_orientations = [
        [item[i] for item in camera_orientations_world] for i in range(3)
    ]  # Flatten each component separately

    # Create actions
    location_action = bpy.data.actions.new("LocationAction")
    rotation_action = bpy.data.actions.new("RotationAction")

    # Create fcurves for each component of location and rotation
    for index in range(
        3
    ):  # There are 3 components for location (x, y, z) and rotation (roll, pitch, yaw)
        location_fcurve = location_action.fcurves.new("location", index=index)
        rotation_fcurve = rotation_action.fcurves.new("rotation_euler", index=index)
        location_fcurve.keyframe_points.add(count=len(frames))
        rotation_fcurve.keyframe_points.add(count=len(frames))

        # Prepare flattened data for this component
        flattened_positions_component = [
            x for co in zip(frames, flattened_positions[index]) for x in co
        ]
        flattened_orientations_component = [
            x for co in zip(frames, flattened_orientations[index]) for x in co
        ]

        # Populate the keyframe points
        location_fcurve.keyframe_points.foreach_set("co", flattened_positions_component)
        rotation_fcurve.keyframe_points.foreach_set("co", flattened_orientations_component)
        location_fcurve.update()
        rotation_fcurve.update()
    print("done creating fcurves")

    # Assign the actions to the camera
    camera.animation_data_create()
    camera.animation_data.action = location_action
    # Add the rotation fcurves to the existing action
    for index in range(3):
        rotation_fcurve = camera.animation_data.action.fcurves.new("rotation_euler", index=index)
        flattened_orientations_component = [
            x for co in zip(frames, flattened_orientations[index]) for x in co
        ]
        rotation_fcurve.keyframe_points.add(count=len(frames))
        rotation_fcurve.keyframe_points.foreach_set("co", flattened_orientations_component)
        rotation_fcurve.update()
    print("done done")


# create_animation("/Users/armandpl/Dev/skyline/trajectory_optimization/data/fixed_speed/trajectories.txt")
animate_camera(
    "/Users/armandpl/Dev/skyline/trajectory_optimization/data/fixed_speed/trajectories.txt"
)
create_augmentations(
    "/Users/armandpl/Dev/skyline/trajectory_optimization/data/fixed_speed/trajectories.txt"
)
