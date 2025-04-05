import logging

import gym
import numpy as np
from gym import spaces, utils

from .cartpole_mapdl_simple import CartPoleMapdlSimple

logger = logging.getLogger(__name__)


class CartPoleEnv(gym.Env, utils.EzPickle):
    metadata = {"render.modes": ["human"]}

    actions = {0: -1, 1: 1}

    def __init__(self):
        self.env = CartPoleMapdlSimple()
        self.status = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-5, -np.inf, -180.0, -np.inf]),
            high=np.array([5, np.inf, 180.0, np.inf]),
            dtype=np.float32,
        )

        self.steps_beyond_done = None

    def __del__(self):
        pass

    def _configure_environment(self):
        pass

    def step(self, action):
        self._take_action(action)
        self.env.step(CartPoleEnv.actions[action])

        self.status = self.env.is_over()
        reward = self._get_reward()
        ob = self.env.get_state()
        episode_over = self.status
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        self.env.act(CartPoleEnv.actions[action])

    def _get_reward(self):
        if not self.env.is_over():
            return 1
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0.0
            return 1
        else:
            self.steps_beyond_done += 1
            return 0

    def reset(self):
        self.steps_beyond_done = None
        self.env.reset()
        return self.env.get_state()

    def _render(self, mode="human", close=False):
        pass
