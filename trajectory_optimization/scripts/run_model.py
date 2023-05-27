import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.colors import TwoSlopeNorm

from trajectory_optimization.utils import load_model_and_instantiate_env

if __name__ == "__main__":
    run = wandb.init(project="skyline", job_type="use_agent")

    # TODO maybe add option to load local model
    model, env = load_model_and_instantiate_env(
        artifact_alias="agent:latest", time_limit=None, render_mode="human", max_wheels_out=4
    )

    trajectory = []

    obs, _ = env.reset()
    while True:
        action, _states = model.predict([obs], deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action[0])

        if terminated:
            if "lap_time" in info.keys():
                break
            else:
                obs, _ = env.reset()
                trajectory = []

        car = env.unwrapped.car
        trajectory.append([car.pos_x, car.pos_y, car.speed])

    trajectory = np.array(trajectory)

    speed = trajectory[:, 2]  # get the speed column

    # compute difference in speed between consecutive elements
    delta_speed = np.diff(speed, prepend=speed[0])

    # add the acceleration column to the numpy array
    trajectory = np.column_stack((trajectory, delta_speed))

    # plot trajectory
    track = env.unwrapped.track

    x = trajectory[:, 0]  # get the x coordinates
    y = trajectory[:, 1]  # get the y coordinates
    acc = trajectory[:, 3]  # get the acceleration

    # Set the colormap
    cmap = plt.get_cmap("RdYlGn")
    norm = TwoSlopeNorm(vmin=min(acc), vcenter=0, vmax=max(acc))

    plt.plot(*track.outer.xy, color="black")
    plt.plot(*track.inner.xy, color="black")

    # Create scatter plot with color mapping to acceleration
    sc = plt.scatter(x, y, c=acc, cmap=cmap, norm=norm, s=10)

    # Create a colorbar
    plt.colorbar(sc, label="Acceleration")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

    env.close()
    run.finish()
