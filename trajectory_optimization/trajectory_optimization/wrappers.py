import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class HistoryWrapper(gym.Wrapper):
    """Track history of observations for given amount of steps Initial steps are zero-filled."""

    def __init__(self, env: gym.Env, steps: int, use_continuity_cost: bool):
        super().__init__(env)
        assert steps > 1, "steps must be > 1"
        self.steps = steps
        self.use_continuity_cost = use_continuity_cost

        # concat obs with action
        self.step_low = np.concatenate([self.observation_space.low, self.action_space.low])
        self.step_high = np.concatenate([self.observation_space.high, self.action_space.high])

        # stack for each step
        obs_low = np.tile(self.step_low, (self.steps, 1))
        obs_high = np.tile(self.step_high, (self.steps, 1))

        self.observation_space = Box(low=obs_low, high=obs_high)

        self.history = self._make_history()

    def _make_history(self):
        return [np.zeros_like(self.step_low) for _ in range(self.steps)]

    def _continuity_cost(self, obs):
        action = obs[-1][-1]
        last_action = obs[-2][-1]
        continuity_cost = np.power((action - last_action), 2).sum()

        return continuity_cost

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)
        obs = np.array(self.history)

        if self.use_continuity_cost:
            continuity_cost = self._continuity_cost(obs)
            reward -= continuity_cost
            info["continuity_cost"] = continuity_cost

        return obs, reward, done, info

    def reset(self):
        self.history = self._make_history()
        self.history.pop(0)
        obs = np.concatenate([self.env.reset(), np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        return np.array(self.history)
