import numpy as np
import cv2

from gym.spaces import Box
from gym import ObservationWrapper


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """
    def __init__(self, env, size):
        super().__init__(env)
        assert size > 0
        self.size = size
        
        shape = (self.size, self.size) + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)
        
    def observation(self, observation):
        observation = cv2.resize(observation, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation


@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
@pytest.mark.parametrize('size', [16, 32])
def test_resize_observation(env_id, size):
    env = gym.make(env_id)
    env = ResizeObservation(env, size)

    assert env.observation_space.shape[-1] == 3
    assert env.observation_space.shape[:2] == (size, size)
    obs = env.reset()
    assert obs.shape == (size, size, 3)
