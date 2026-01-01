import gymnasium as gym
from gymnasium import spaces
import h5py
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from collections import OrderedDict

import gymnasium as gym
import numpy as np
from collections import deque

class WindowBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, window_size):
        super().__init__(env)
        self.window_size = window_size
        
        # Internal buffer to store history of observations
        # and a counter to track valid timesteps for the mask
        self.obs_buffer = deque(maxlen=window_size)
        self.steps_filled = 0

        # Update Observation Space
        new_spaces = {}
        for key, space in self.observation_space.spaces.items():
            new_shape = (window_size,) + space.shape
            new_spaces[key] = gym.spaces.Box(
                low=np.min(space.low), 
                high=np.max(space.high), 
                shape=new_shape, 
                dtype=space.dtype
            )
        
        # Add the pad mask: True for real data, False for padding
        new_spaces['timestep_pad_mask'] = gym.spaces.Box(
            low=0, high=1, shape=(window_size,), dtype=bool
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset history: Fill with the first observation
        self.obs_buffer.clear()
        for _ in range(self.window_size):
            self.obs_buffer.append(obs)
        
        # At reset, only the last element is "truly" the result of a step
        # though usually, the first frame is considered valid.
        self.steps_filled = 1 
        
        return self._transform_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.obs_buffer.append(obs)
        self.steps_filled = min(self.window_size, self.steps_filled + 1)
        
        return self._transform_obs(), reward, terminated, truncated, info

    def _transform_obs(self):
        """Packages the deque into the windowed dictionary format."""
        batch_obs = {}
        
        # 1. Create the mask: [False, False, True, True, True]
        mask = np.zeros(self.window_size, dtype=bool)
        # The last 'steps_filled' elements are valid
        mask[-self.steps_filled:] = True
        
        batch_obs['timestep_pad_mask'] = mask

        # 2. Stack the observations for each key
        for key in self.env.observation_space.spaces.keys():
            batch_obs[key] = np.stack([step[key] for step in self.obs_buffer], axis=0)
            
        return batch_obs

class TiagoGym(gym.Env):
    def __init__(self, data_dir, demo_index=0):
        super().__init__()
        self.hdf5_object = h5py.File(data_dir, 'r')
        self.demo_index = demo_index
        if 'data' in self.hdf5_object:
            self.num_samples = self.hdf5_object[f'data/demo_{self.demo_index}/obs/tiago_head_image'].shape[0]
        else:
            self.num_samples = self.hdf5_object[f'obs/tiago_head_image'].shape[0]
        self.curr_index = -1 
        self.h5_prefix = f'data/demo_{self.demo_index}/' if 'data' in self.hdf5_object else ''

        # 1. Be very explicit with shapes and dtypes here
        self.observation_space = spaces.Dict({
            'left': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(8,), 
                dtype=np.float32
            ),
            'tiago_head_image': spaces.Box(
                low=0.0, 
                high=255.0, 
                shape=(480, 480, 3), 
                dtype=np.float16
            )
        })
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def _observation(self):
        # 2. Use .reshape() and .astype() to guarantee the shape matches the space exactly
        left_data = self.hdf5_object[f'{self.h5_prefix}obs/left'][self.curr_index]
        image_data = self.hdf5_object[f'{self.h5_prefix}obs/tiago_head_image'][self.curr_index]
       
        #import pdb; pdb.set_trace()
        return {
            'left': np.array(left_data, dtype=np.float64).reshape(8,),
            'tiago_head_image': np.array(image_data, dtype=np.float16).reshape(480, 480, 3)
        }

    def step(self, action):
        # Gymnasium step returns: obs, reward, terminated, truncated, info
        self.curr_index += 1
        
        # 'terminated' usually means the goal/fail state is reached
        # 'truncated' usually means a time limit or end of data is reached
        terminated = False 
       #if self.curr_index >= self.num_samples - 1:
            #import pdb; pdb.set_trace()
        truncated = (self.curr_index >= self.num_samples - 1)
        
        reward = 0.0
        info = {}
        
        return self._observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Gymnasium reset must handle seed and return (obs, info)
        super().reset(seed=seed)
        
        self.curr_index = -1
        obs = self._observation()
        info = {}
        
        return obs, info

    def close(self):
        if hasattr(self, 'hdf5_object'):
            self.hdf5_object.close()

if __name__ == "__main__":

    env = TiagoGym(data_dir='/home/ec2-user/dec_21_processed/processed_il_data/processed_il_data/data_0_to_8.h5', demo_index=0)
    env = WindowBufferWrapper(env, window_size=5)
    obs = env.reset()
    done = False
    while not done:
     

        obs, reward, _, done, info = env.step(action)
        print(obs['timestep_pad_mask'])
    env.close()