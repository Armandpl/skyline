import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from trajectory_optimization.car_env import CarRacing
from trajectory_optimization.wrappers import HistoryWrapper


def main():
    env = CarRacing(random_init=True)
    env = HistoryWrapper(env=env, steps=2, use_continuity_cost=True)

    # add monitoring
    # add video recording

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    # Save the agent
    model.save("car_test")

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
