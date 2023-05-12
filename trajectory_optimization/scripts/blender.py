# copy paste into blender
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
    material = bpy.data.materials.new(name="White Material")
    material.diffuse_color = (1, 1, 1, 1)  # RGBA white color

    # Assign the material to the curve
    curve_object.data.materials.append(material)

    # flatten the tube, make it a strip
    curve_object.scale = (1, 1, 0.01)


# plot_line("/Users/armandpl/Dev/skyline/trajectory_optimization/notebooks/inner.txt", "inner")
# plot_line("/Users/armandpl/Dev/skyline/trajectory_optimization/notebooks/outer.txt", "outer")


def create_animation(traj_path):
    traj = np.loadtxt(traj_path)  # [x, y, yaw]

    # Assume you have lists of car positions and orientations
    car_positions = traj[:, 0:2]  # List of Vector((x, y)) for each frame
    car_positions = np.pad(car_positions, ((0, 0), (0, 1)))  # add z=0
    car_orientations = traj[:, 2].reshape(-1, 1)  # List of Euler(yaw) for each frame
    # car_orientations = np.radians(car_orientations)
    car_orientations = np.pad(car_orientations, ((0, 0), (2, 0)))  # add pitch = 0, roll = 0

    # Camera offset from car
    camera_offset_position = mathutils.Vector((0.105, 0, 0.134))  # x, y, z
    camera_offset_rotation = mathutils.Euler(
        (np.radians(90) + np.radians(-12), np.radians(0), np.radians(-90))
    )

    # Define camera FOV
    camera_fov = np.radians(160)  # In radians TODO set it in mm instead? 3.15mm
    # set sensor size horizontal 3.6mm?

    # Create a new camera object
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.data.angle = camera_fov

    # Set camera to be the active camera
    bpy.context.scene.camera = camera

    # Make sure you have as many frames as you have positions and orientations
    bpy.context.scene.frame_end = len(car_positions)

    # Iterate over each frame
    for frame in range(len(car_positions)):
        # Set the current frame
        bpy.context.scene.frame_set(frame)

        # Create rotation matrix from car's orientation
        car_rot_matrix = mathutils.Matrix.Rotation(
            car_orientations[frame][2], 4, "Z"
        )  # 4 for 4x4 matrix, 'Z' for rotation about Z-axis

        # Transform camera offset from car's coordinate frame to world coordinate frame
        camera_offset_position_world = car_rot_matrix @ camera_offset_position

        # Compute camera position and rotation based on car's data and transformed offset
        camera.location = car_positions[frame] + camera_offset_position_world
        camera.rotation_euler = car_orientations[frame] + camera_offset_rotation

        # Add keyframes for these properties
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Render the animation
    # bpy.ops.render.render(animation=True, write_still=True)


create_animation(
    "/Users/armandpl/Dev/skyline/trajectory_optimization/data/centerline_trajectories.txt"
)
