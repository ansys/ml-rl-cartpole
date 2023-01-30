import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils
import logging
import numpy as np
from . import CartPoleMapdlSimple


logger = logging.getLogger(__name__)


class CartPoleEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    actions = {0: -1, 1: 1}

    def __init__(self):
        self.env = CartPoleMapdlSimple()
        self.status = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([-5, -np.inf, -180., -np.inf]),
                                            high=np.array([5, np.inf, 180., np.inf]),
                                            dtype=np.double)

        self.steps_beyond_done = None

    def __del__(self):
        pass 

    def _configure_environment(self):
        pass

    def step(self, action):
        self._take_action(action)
        self.env.step(CartPoleEnv.actions[action])

        done, _ = self.env.is_over()
        self.status = done
        reward = self._get_reward()
        ob = self.env.get_state()
        episode_over = self.status
        return ob, reward, episode_over, False, {}

    def _take_action(self, action):
        self.env.act(CartPoleEnv.actions[action])

    def _get_reward(self):
        done, _  = self.env.is_over()
        if not done:
            return 1
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0.
            return 1
        else:
            self.steps_beyond_done += 1
            return 0

    def reset(self):
        self.steps_beyond_done = None
        self.env.reset()
        return self.env.get_state(), {}

    def _render(self, mode='human', close=False):
        pass
