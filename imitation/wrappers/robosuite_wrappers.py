import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
from collections import OrderedDict
from robosuite.wrappers import Wrapper

class RobosuiteImageFlipWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        
    @property
    def observation_space(self):
        ob_space = OrderedDict()
        for key in self.obs_keys:
            ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=self.modality_dims[key])
        return spaces.Dict(ob_space)
    
    def flip_imgs(self, obs):
        ob = {}
        for k in obs:
            if 'image' in k:
                ob[k] = np.flip(obs[k], axis=0)
            else:
                ob[k] = obs[k]
        return ob
    
    def check_success(self):
        return self.env._check_success()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.flip_imgs(obs)
    
    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        return self.flip_imgs(obs), reward, terminated, info

class RobosuiteGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None

    def __init__(self, env):
        super().__init__(env=env)
        self.name = type(self.env).__name__

        self.obs_keys = []
        self.env.spec = None

        obs = self.env.reset()
        for k in obs.keys():
            self.obs_keys.append(k)

        self.modality_dims = {key: obs[key].shape for key in self.obs_keys}

        if self.env.action_space is not None:
            self.action_space = self.env.action_space
        else:
            low, high = self.env.action_spec
            self.action_space = spaces.Box(low, high)

    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        for key in self.obs_keys:
            ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=self.modality_dims[key])
        return spaces.Dict(ob_space)
    
    def filter_obs(self, obs):
        return {key: obs[key] for key in self.obs_keys}
    
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("seed must be an integer")
        return self.filter_obs(self.env.reset()), {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # if self.env._check_success():
        #     print('Gym Wrapper - Success!')
        #     done = True
        return self.filter_obs(obs), reward, done, False, info

class FrameStackWrapper(gym.core.Wrapper):

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = {k: [] for k in self.env.observation_space.spaces.keys()}
        
    @property
    def observation_space(self):
        ob_space = OrderedDict()
        for key in self.frames.keys():
            space = self.env.observation_space.spaces[key]
            if len(space.shape) == 1:
                ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stack*space.shape[0],))
            elif len(space.shape) == 3:
                ob_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(space.shape[0], space.shape[1], self.num_stack*space.shape[2]))
            else:
                raise ValueError('Invalid shape', key, space)
        return spaces.Dict(ob_space)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for k in self.frames.keys():
            self.frames[k] = [obs[k] for _ in range(self.num_stack)]
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._append_frame(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _append_frame(self, obs):
        for k in self.frames.keys():
            self.frames[k].append(obs[k])
            if len(self.frames[k]) > self.num_stack:
                self.frames[k].pop(0)

    def _get_obs(self):
        # return {k: np.concatenate(self.frames[k], axis=-1) for k in self.frames.keys()}
        obs = {}
        for k in self.frames.keys():
            obs[k] = np.concatenate(self.frames[k], axis=-1)
            # if len(space.shape) == 1:
            #     obs[k] = np.concatenate(self.frames[k], axis=0)
            # elif len(space.shape) == 3:
            #     obs[k] = 
            # else:
            #     raise ValueError('Invalid shape', k, space)
        return obs
    
class RobosuiteDiscreteControlWrapper(gym.core.Wrapper):
    '''
        Maps discrete integer actions into continuous list of actions.
        If a list action is provided then it sends it directly to the environment without processing.
    '''

    def __init__(self, env):
        super().__init__(env=env)

        self.action_map = {
            0: np.array([0, 0, 0, 0]),
            1: np.array([1, 0, 0, 0]),
            2: np.array([-1, 0, 0, 0]),
            3: np.array([0, 1, 0, 0]),
            4: np.array([0, -1, 0, 0]),
            5: np.array([0, 0, 1, 0]),
            6: np.array([0, 0, -1, 0]),
            7: np.array([0, 0, 0, 1]), # close gripper
            8: np.array([0, 0, 0, -1]) # open gripper
        }

        self.action_dim = len(list(self.action_map.keys()))
        self.max_action = 0.5

    @property
    def action_space(self):
        return spaces.Discrete(self.action_dim)

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            return self.env.step(action)
        action_id = action
        action = self.action_map[int(action)]
        if action_id in list(range(1, 7)):
            action = self.max_action * action / np.linalg.norm(action)
        return self.env.step(action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

class ReachButtonRewardWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.max_steps = 300
        self.step_count = 0
    
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            trunc = True
        
        reward = 0

        from_xyz = obs['robot0_eef_pos']
        to_xyz = obs['Button1_pos'] + np.array([0, 0, 0.1])

        dist = np.linalg.norm(from_xyz - to_xyz)
        reward += (0.1 - dist)
        
        # if dist < 0.1:
        #     reward += 0.3

        if self.env.unwrapped.buttons_on[1]:
            reward += 70
            done = True
        return obs, reward, done, trunc, info
    
    def reset(self, seed=None):
        self.step_count = 0
        return self.env.reset(seed=seed)