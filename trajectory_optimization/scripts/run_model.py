import stable_baselines3 as sb3
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from trajectory_optimization.car_env import CarRacing
from trajectory_optimization.wrappers import HistoryWrapper

if __name__ == "__main__":

    def make_env():
        env = CarRacing(random_init=True, render_mode="human")
        env = HistoryWrapper(env=env, steps=2, use_continuity_cost=False)
        return env

    vec_env = DummyVecEnv([make_env])

    model = SAC.load("car_test")

    # Enjoy trained agent
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
