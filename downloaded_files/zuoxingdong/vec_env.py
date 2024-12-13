import numpy as np

from lagom.vis import GridImage

try:  # workaround on server without fake screen but still running other things well
    from lagom.vis import ImageViewer
except ImportError:
    pass


class VecEnv(object):
    r"""A vectorized environment runs serially for each sub-environment. 
    
    Each observation returned from vectorized environment is a batch of observations 
    for each sub-environment. And :meth:`step` is expected to receive a batch of 
    actions for each sub-environment. 
    
    .. note::
    
        All sub-environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported. 
    
    Args:
        list_make_env (list): a list of functions each returns an instantiated enviroment. 
        observation_space (Space): observation space of the environment
        action_space (Space): action space of the environment
        
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    closed = False
    viewer = None

    def __init__(self, list_make_env):
        self.list_make_env = list_make_env
        self.list_env = [make_env() for make_env in list_make_env]
        self.observation_space = self.list_env[0].observation_space
        self.action_space = self.list_env[0].action_space
        self.reward_range = self.list_env[0].reward_range
        self.spec = self.list_env[0].spec

    def step(self, actions):
        r"""Ask all the environments to take a step with a list of actions, each for one environment. 
        
        Args:
            actions (list): a list of actions, each for one environment. 
            
        Returns:
            tuple: a tuple of (observations, rewards, dones, infos)
                * observations (list): a list of observations, each returned from one environment 
                  after executing the given action. 
                * rewards (list): a list of scalar rewards, each returned from one environment. 
                * dones (list): a list of booleans indicating whether the episode terminates, each 
                  returned from one environment. 
                * infos (list): a list of dictionaries of additional informations, each returned 
                  from one environment. 
        """
        assert len(actions) == len(self)
        observations = []
        rewards = []
        dones = []
        infos = []
        for i, (env, action) in enumerate(zip(self.list_env, actions)):
            observation, reward, done, info = env.step(action)
            # If done=True, reset environment, store last observation in info and report new initial observation
            if done:
                info['last_observation'] = observation
                observation = env.reset()
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos
    
    def reset(self):
        r"""Reset all the environments and return a list of initial observations from each environment. 
        
        .. warning::
        
            If :meth:`step_async` is still working, then it will be aborted. 
        
        Returns:
            list: a list of initial observations from all environments. 
        """
        observations = [env.reset() for env in self.list_env]
        return observations
    
    def render(self, mode='human'):
        r"""Render all the environments. 
        
        It firstly retrieve RGB images from all environments and use :class:`GridImage`
        to make a grid of them as a single image. Then it either returns the image array
        or display the image to the screen by using :class:`ImageViewer`. 
        
        See docstring in :class:`Env` for more detais about rendering. 
        """
        # Get images from all environments with shape [N, H, W, C]
        imgs = self.get_images()
        imgs = np.stack(imgs)
        # Make a grid of images
        grid = GridImage(ncol=5, padding=5, pad_value=0)
        imgs = imgs.transpose(0, 3, 1, 2)  # to shape [N, C, H, W]
        grid.add(imgs)
        gridimg = np.asarray(grid())
        gridimg = gridimg.transpose(0, 2, 3, 1)  # back to shape [N, H, W, C]
        
        # render the grid of image
        if mode == 'human':
            self.get_viewer()(gridimg)
        elif mode == 'rgb_array':
            return gridimg
        else:
            raise ValueError(f'expected human or rgb_array, got {mode}')

    def get_images(self):
        r"""Returns a batched RGB array with shape [N, H, W, C] from all environments. 
        
        Returns:
            ndarray: a batched RGB array with shape [N, H, W, C]    
        """
        return [env.render(mode='rgb_array') for env in self.list_env]
    
    def get_viewer(self):
        r"""Returns an instantiated :class:`ImageViewer`. 
        
        Returns:
            ImageViewer: an image viewer
        """
        if self.viewer is None:  # create viewer is not existed
            self.viewer = ImageViewer(max_width=500)  # set a max width here
        return self.viewer
    
    def close_extras(self):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        return [env.close() for env in self.list_env]
    
    def close(self):
        r"""Close all environments. 
        
        It closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``. 
        
        .. warning::
        
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is useful for parallelized environments. 
        
        .. note::
        
            This will be automatically called when garbage collected or program exited. 
            
        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True
    
    @property
    def unwrapped(self):
        r"""Unwrap this vectorized environment. 
        
        Useful for sequential wrappers applied, it can access information from the original 
        vectorized environment. 
        """
        return self
    
    def __len__(self):
        return len(self.list_make_env)
    
    def __getitem__(self, index):
        return self.list_env[index]
    
    def __setitem__(self, index, x):
        self.list_env[index] = x
    
    def __repr__(self):
        return f'<{self.__class__.__name__}: {len(self)}, {self.spec.id}>'
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False 

    
class VecEnvWrapper(VecEnv):
    r"""Wraps the vectorized environment to allow a modular transformation. 
    
    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code. 
    
    .. note::
    
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    
    """
    def __init__(self, env):
        assert isinstance(env, VecEnv)
        self.env = env
        self.metadata = env.metadata
        
        self.list_make_env = env.list_make_env
        self.list_env = env.list_env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.spec = env.spec
        
    def step(self, actions):
        return self.env.step(actions)
    
    def reset(self):
        return self.env.reset()
    
    def get_images(self):
        return self.env.get_images()
    
    def close_extras(self):
        return self.env.close_extras()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def __len__(self):
        return len(self.env)
    
    def __getitem__(self, index):
        return self.env[index]
    
    def __setitem__(self, index, x):
        self.env[index] = x
    
    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.env}>'

    
    
    
    
    
    
@pytest.mark.parametrize('env_id', ['CartPole-v0', 'Pendulum-v0'])
@pytest.mark.parametrize('num_env', [1, 3, 5])
def test_vec_env(env_id, num_env):
    def make_env():
        return gym.make(env_id)
    base_env = make_env()
    list_make_env = [make_env for _ in range(num_env)]
    env = VecEnv(list_make_env)
    assert isinstance(env, VecEnv)
    assert len(env) == num_env
    assert len(list(env)) == num_env
    assert env.observation_space == base_env.observation_space
    assert env.action_space == base_env.action_space
    assert env.reward_range == base_env.reward_range
    assert env.spec.id == base_env.spec.id
    obs = env.reset()
    assert isinstance(obs, list) and len(obs) == num_env
    assert all([x in env.observation_space for x in obs])
    actions = [env.action_space.sample() for _ in range(num_env)]
    observations, rewards, dones, infos = env.step(actions)
    assert isinstance(observations, list) and len(observations) == num_env
    assert isinstance(rewards, list) and len(rewards) == num_env
    assert isinstance(dones, list) and len(dones) == num_env
    assert isinstance(infos, list) and len(infos) == num_env
    env.close()
    assert env.closed