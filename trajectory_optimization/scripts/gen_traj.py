import numpy as np
import wandb
from stable_baselines3 import SAC

from trajectory_optimization import data_dir
from trajectory_optimization.utils import load_model_and_instantiate_env

if __name__ == "__main__":
    NB_STEPS = 10_000
    TRAJ_LEN = 45
    NB_EPISODES = NB_STEPS // TRAJ_LEN

    run = wandb.init(project="skyline", job_type="gen_traj")
    model, env = load_model_and_instantiate_env(
        artifact_alias="agent:latest", time_limit=TRAJ_LEN, max_wheels_out=4, render_mode="human"
    )

    # trajectory at each step should contain
    # pos_x, pos_y, yaw, speed, steering_command in deg, speed_command in m/s
    trajectories = []

    obs, truncated, terminated = None, False, False
    current_traj = []
    while len(trajectories) < NB_EPISODES:
        if obs is None or (truncated or terminated):
            obs, _ = env.reset()
            if truncated:  # means the car successfully ran TRAJ_LEN steps without crashing
                trajectories.append(current_traj)
                print(f"{len(trajectories)} / {NB_EPISODES}")
            current_traj = []

        action, _states = model.predict([obs], deterministic=True)
        action = action[0]  # action[0] bc model was trained on vec env
        obs, reward, terminated, truncated, info = env.step(action)

        car = env.unwrapped.car
        rescaled_action = info["rescaled_action"]
        done = terminated or truncated
        traj_step = [
            car.pos_x,
            car.pos_y,
            car.yaw,
            car.speed,
            rescaled_action[
                0
            ],  # note this is the action that caused this state, not the action to take for this state!
            None,  # no speed command bc fixed speed for now
            done,
        ]
        current_traj.append(traj_step)

    trajectories = np.array(trajectories)
    # Reshape the 3D trajectories (n_traj, n_steps, traj_step_len) array to a 2D array (n_traj*n_step, traj_step_len)
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    np.savetxt("../data/fixed_speed_rl_trajectories.txt", trajectories)

    # TODO save to wandb artifacts
    run.finish()
