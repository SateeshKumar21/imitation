import torch
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from imitation.algo.dynamics_model import ReconstructionInverseDynamicsModel

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    import h5py
    f = h5py.File(args.data_path, 'r+')
    
    # Load the module
    model = ReconstructionInverseDynamicsModel.load_weights(args.ckpt)

    for demo in f['data'].keys():
        print(demo)
        obs = {}
        for k in f['data'][demo]['obs'].keys():
            obs[k] = torch.from_numpy(f['data'][demo]['obs'][k][:]).to(model.device).float()
        
        original_obs = deepcopy(obs)

        z = model.nets["obs_encoder"](obs)

        
        if 'obs_embeddings' in f['data'][demo]['obs'].keys():
            del f['data'][demo]['obs']['obs_embeddings']
        f['data'][demo]['obs'].create_dataset('obs_embeddings', data=z.detach().numpy())

        # decoded_obs = model.reconstruct_obs(z)

        # eef_pos_t = original_obs['robot0_eef_pos']
        # # eef_pos_tn = model.nets['normalizer'].normalize_by_key(obs['robot0_eef_pos'], 'robot0_eef_pos')
        
        # decoded_obs = decoded_obs['robot0_eef_pos']
        # eef_pos_p = decoded_obs.detach().numpy()
        # # eef_pos_pn = model.nets['normalizer'].denormalize_by_key(decoded_obs, 'robot0_eef_pos').detach().numpy()
        # # eef_pos_pn = model.nets['normalizer'].normalize_by_key(eef_pos_pn, 'robot0_eef_pos').detach().numpy()

        # ax = plt.axes(projection='3d')
        # ax.scatter3D(eef_pos_t[:, 0], eef_pos_t[:, 1], eef_pos_t[:, 2], c='b')
        # ax.scatter3D(eef_pos_p[:, 0], eef_pos_p[:, 1], eef_pos_p[:, 2], c='r')
        # # ax.scatter3D(eef_pos_tn[:, 0], eef_pos_tn[:, 1], eef_pos_tn[:, 2], c='g')
        # # ax.scatter3D(eef_pos_pn[:, 0], eef_pos_pn[:, 1], eef_pos_pn[:, 2], c='y')
        # plt.show()
        
    f.close()
