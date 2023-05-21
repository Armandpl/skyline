import numpy as np
from gymnasium.wrappers import TimeLimit
from simple_pid import PID
from stable_baselines3.common.noise import NormalActionNoise

from trajectory_optimization.car_env import CarRacing

if __name__ == "__main__":
    NB_FRAMES = 10_000
    a = np.array([0.0, 0.25])

    env = CarRacing(random_init=True)
    env = TimeLimit(env, max_episode_steps=100)

    action_noise = NormalActionNoise(mean=np.array([0.0]), sigma=np.array([0.2]))

    # TODO save trajectories and actions

    pid = PID(10, 0, 4.5, setpoint=0)

    trajectories = None
    while trajectories is None or trajectories.shape[0] < NB_FRAMES:
        env.reset()
        states = []  # list of
        while True:
            s, r, terminated, truncated, info = env.step(a)

            distance = info["distance_to_centerline"]
            action = pid(distance)
            action = action + action_noise()
            action = np.clip(action, -1, 1)
            a[0] = action[0]
            car = env.unwrapped.car
            states.append([car.pos_x, car.pos_y, car.yaw, action[0]])
            # print(f"distance: {distance}\naction: {action}\n")

            if terminated or truncated:
                if (
                    truncated
                ):  # truncated = car got back to following centerline and didn't hit wall
                    states = np.array(states)
                    if trajectories is None:
                        trajectories = states
                    else:
                        trajectories = np.concatenate((trajectories, np.array(states)), axis=0)
                    print(f"{trajectories.shape[0]} / {NB_FRAMES}")
                break
    env.close()

    np.savetxt("../data/centerline_trajectories.txt", trajectories)
