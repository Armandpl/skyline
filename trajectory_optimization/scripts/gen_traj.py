import numpy as np
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from train import make_env

from trajectory_optimization import data_dir

if __name__ == "__main__":
    NB_STEPS = 10_000
    TRAJ_LEN = 100
    NB_EPISODES = NB_STEPS // TRAJ_LEN

    env = TimeLimit(make_env(), max_episode_steps=TRAJ_LEN)

    # TODO download from artifacts
    model = SAC.load("car_test")

    # trajectory at each step should contain
    # pos_x, pos_y, yaw, steering_command in deg, speed_command in m/s, speed in m/s
    trajectories = []

    obs, truncated, terminated = None, False, False
    current_traj = []
    while len(trajectories) < NB_EPISODES:
        if obs is None or (truncated or terminated):
            obs, _ = env.reset()
            if truncated:  # means the car successfully ran the 200 steps
                trajectories.append(current_traj)
                print(f"{len(trajectories)} / {NB_EPISODES}")
            current_traj = []

        action, _states = model.predict([obs], deterministic=True)
        action = action[0]  # action[0] bc model was trained on vec env
        obs, reward, terminated, truncated, info = env.step(action)
        car = env.unwrapped.car
        denormed_action = car.denormalize_action(action)
        done = terminated or truncated
        traj_step = [
            car.pos_x,
            car.pos_y,
            car.yaw,
            car.speed,
            denormed_action[0],
            denormed_action[1],
            done,
        ]
        current_traj.append(traj_step)

    trajectories = np.array(trajectories)
    # Reshape the 3D trajectories array to a 2D array
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    np.savetxt("../data/rl_trajectories.txt", trajectories)

    # TODO save to wandb artifacts
