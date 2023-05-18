import stable_baselines3 as sb3
import wandb
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

from trajectory_optimization.car_env import CarRacing
from trajectory_optimization.wrappers import HistoryWrapper


def main():
    env = CarRacing(random_init=True, crash_penalty_weight=0)
    env = HistoryWrapper(env=env, steps=2, use_continuity_cost=False)
    env = TimeLimit(env, max_episode_steps=500)

    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    # TODO
    # add monitoring and wandb logging
    # add video recording
    # check env

    # Instantiate the agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        use_sde=True,  # make sense since we're simulating a robot
        gamma=0.98,  # if we discount future rewards too much the car doesn't care about crashing in the future
    )
    model.load("car_test")

    try:
        # Train the agent and display a progress bar
        model.learn(total_timesteps=int(1e6), progress_bar=True)
    except KeyboardInterrupt:
        pass
    # Save the agent
    model.save("car_test")

    run.finish()
    # TODO Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


if __name__ == "__main__":
    main()
