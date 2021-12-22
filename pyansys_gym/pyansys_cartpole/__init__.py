import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='pyansys-CartPole-v0',
    entry_point='pyansys_cartpole.envs:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

