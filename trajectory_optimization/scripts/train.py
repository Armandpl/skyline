import stable_baselines3 as sb3
import wandb
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from trajectory_optimization.car_env import CarRacing
from trajectory_optimization.wrappers import HistoryWrapper


def make_env():
    env = CarRacing(random_init=True, crash_penalty_weight=0, render_mode="rgb_array")
    check_env(env)
    env = HistoryWrapper(env=env, steps=2, use_continuity_cost=False)
    # env = TimeLimit(env, max_episode_steps=500)
    env = Monitor(env)
    check_env(env)
    return env


class LapTimeCallback(BaseCallback):
    def _on_step(self) -> bool:
        lap_time = self.locals["infos"][0]["lap_time"]  # [0] to access first env of vec env
        if lap_time is not None:
            self.logger.record("lap_time", lap_time)
        return True


def main():
    MAX_STEPS = int(2e6)
    VIDEO_FREQ = MAX_STEPS // 10  # 10 video per training

    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    vec_env = DummyVecEnv([make_env])

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(
        vec_env,
        f"runs/{run.id}/videos",
        record_video_trigger=lambda x: x % VIDEO_FREQ == 0,
        video_length=500,
    )

    # Instantiate the agent
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        use_sde=True,  # make sense since we're simulating a robot
        gamma=0.98,  # if we discount future rewards too much the car doesn't care about crashing in the future
    )
    # model.load("car_test")

    try:
        # Train the agent and display a progress bar
        model.learn(total_timesteps=MAX_STEPS, progress_bar=True, callback=LapTimeCallback())
    except KeyboardInterrupt:
        pass
    # Save the agent
    # TODO save to wandb
    model.save("car_test")

    run.finish()
    # TODO Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


if __name__ == "__main__":
    main()
