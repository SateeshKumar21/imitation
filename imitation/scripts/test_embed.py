import numpy as np
from imitation.algo.bc_rnn import BC_RNN
from imitation.algo.diffusion_policy import DiffusionPolicy
from imitation.algo.bc_transformer import BCTransformer
from imitation.algo.dynamics_model import ReconstructionInverseDynamicsModel
import torch
import cv2
import copy
from imitation.utils.vis_utils import get_saliency_maps, write_video

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dyn_ckpt', type=str, default=None)
    args = parser.parse_args()

    # Load the module
    model = BCTransformer.load_weights(args.ckpt)
    # model = DiffusionPolicy.load_weights(args.ckpt)
    # model = BC_RNN.load_weights(args.ckpt)

    dynamics_model = ReconstructionInverseDynamicsModel.load_weights(args.dyn_ckpt)

    import robosuite as suite
    from l2l.config.env.robosuite.color_lift import env_config
    # initialize the task
    env = suite.make(
        **env_config
    )
    obs = env.reset()

    # Get action limits
    low, high = env.action_spec

    # do visualization
    steps = 0

    video = []
    count_successes = 0
    for i in range(10000):
        original_obs = copy.deepcopy(obs)
        obs = dynamics_model.preprocess_obs(obs)
        original_obs['obs_embeddings'] = dynamics_model.get_embeddings(obs)[0].detach().numpy()
        action = model.get_action(original_obs)#np.random.uniform(low, high)

        env.render()
        obs, reward, done, _ = env.step(action)
        
        for k in obs:
            if 'image' in k:
                obs[k] = obs[k].astype(np.uint8)

        steps += 1

        if done or steps%120==0:
            obs = env.reset()
            steps = 0
            model.reset()

            count_successes += int(done)

            if i > 1200:
                break
    
    print(count_successes)
    # write_video(video, "moving_cam_pnp_with_saliency.mp4")