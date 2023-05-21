import stable_baselines3 as sb3
import wandb
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from train import make_env
from wandb.integration.sb3 import WandbCallback

from trajectory_optimization.car_env import CarRacing
from trajectory_optimization.wrappers import HistoryWrapper

if __name__ == "__main__":
    vec_env = DummyVecEnv([lambda: make_env(render_mode="human")])

    # TODO download from artifacts
    model = SAC.load("car_test")

    # Enjoy trained agent
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
