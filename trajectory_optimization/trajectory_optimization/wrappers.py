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
        obs_low = np.tile(self.step_low, self.steps)
        obs_high = np.tile(self.step_high, self.steps)

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
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.pop(0)

        obs = np.concatenate([obs, action])
        self.history.append(obs)
        obs = np.array(self.history)
        obs = obs.flatten()

        if self.use_continuity_cost:
            continuity_cost = self._continuity_cost(obs)
            reward -= continuity_cost
            info["continuity_cost"] = continuity_cost

        return obs, reward, terminated, truncated, info

    def reset(self):
        self.history = self._make_history()
        self.history.pop(0)
        obs = np.concatenate([self.env.reset()[0], np.zeros_like(self.env.action_space.low)])
        self.history.append(obs)
        return np.array(self.history).flatten(), {}


class RescaleWrapper(gym.Wrapper):
    """Linearly rescale observation and action space between -1 and 1."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=np.zeros_like(self.env.observation_space.low) - 1,
            high=np.zeros_like(self.env.observation_space.low) + 1,
        )
        self.action_space = Box(
            low=np.zeros_like(self.env.action_space.low) - 1,
            high=np.zeros_like(self.env.action_space.low) + 1,
        )

    def rescale_to_minus_plus_one(self, value, low, high):
        # rescale from original range to [-1, 1]
        value = np.clip(value, low, high)
        range_old = high - low
        range_new = 2.0  # as the range is now -1 to 1

        # apply the transformation
        rescaled_value = -1 + ((value - low) / range_old) * range_new
        return rescaled_value

    def rescale_from_minus_plus_one(self, value, low, high):
        # rescale from [-1, 1] to original range
        value = np.clip(value, -1, 1)
        range_old = 2.0  # as the range is now -1 to 1
        range_new = high - low

        # apply the transformation
        rescaled_value = low + ((value + 1) / range_old) * range_new
        return rescaled_value

    def step(self, action):
        # Rescale action, from [-1,1] to original action_space
        action = self.rescale_from_minus_plus_one(
            action, self.env.action_space.low, self.env.action_space.high
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["rescaled_action"] = action
        # Rescale observation, from original observation_space to [-1,1]
        obs = self.rescale_to_minus_plus_one(
            obs, self.env.observation_space.low, self.env.observation_space.high
        )
        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        obs = self.rescale_from_minus_plus_one(
            obs, self.env.observation_space.low, self.env.observation_space.high
        )
        return obs, info
