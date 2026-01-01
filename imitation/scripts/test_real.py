from imitation.algo.diffusion_policy import DiffusionPolicy
import torch
from imitation.envs.real_env import TiagoGym, WindowBufferWrapper
import pickle as pkl
import cv2
import numpy as np

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    #data_dir = '/home/ec2-user/dec_21_processed/processed_il_data/processed_il_data/data_0_to_8.h5'
    data_dir = "/home/ec2-user/dec_21_processed/original_il_data/original_il_data/demo_0.h5"

    env = TiagoGym(data_dir=data_dir, demo_index=0)
   # env = WindowBufferWrapper(env, window_size=5)
    env_obs, _ = env.reset()

    model = DiffusionPolicy.load_weights(args.ckpt)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    obs = pkl.load(open("test_data.pkl", 'rb'))
    #action = model.get_action_debug(obs['obs'], batched=False) 
    #action = model.get_action(env_obs, batched=False)
    for i in range(15):
        obs, _, done, _, _ = env.step(env.action_space.sample())
       
        image = obs['tiago_head_image']
      
        image = image.astype(np.float32) 
        obs['tiago_head_image'] = cv2.resize(image, (224, 224))
        print(obs['tiago_head_image'].shape)
    
        #import pdb; pdb.set_trace()
        action = model.get_action(obs, batched=False)

        print(action)
        print("GT action:  ", env.hdf5_object[f'actions'][env.curr_index][3:])

#    print("loaded model")