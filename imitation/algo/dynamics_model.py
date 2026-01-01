import numpy as np
import torch
import torch.nn as nn

from imitation.algo.base_algo import BaseAlgo
from imitation.utils.general_utils import AttrDict
from imitation.models.normalizers import DictNormalizer
from imitation.models.obs_nets import ObservationEncoder, ObservationDecoder
from imitation.models.base_nets import MLP
from imitation.models.distribution_nets import Gaussian

from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from imitation.utils.file_utils import get_obs_key_to_modality_from_config

class ReconstructionDynamicsModel(BaseAlgo):

    def __init__(self, config):
        super(ReconstructionDynamicsModel, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes

        self.low_dim_obs_keys = observation_config.obs['low_dim']
        self.rgb_obs_keys = observation_config.obs['rgb']
        
        action_dim = keys_to_shapes['ac_dim']

        self.nets = nn.ModuleDict()
        key_to_norm_type = observation_config.obs_keys_to_normalize
        key_to_norm_type['actions'] = config.policy_config.action_normalization_type
        normalizer = DictNormalizer(config.normalization_stats, key_to_norm_type=key_to_norm_type)
        self.nets["normalizer"] = normalizer
        
        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        self.nets["obs_encoder"] = nn.Sequential(
            obs_encoder,
            MLP(
                input_dim=obs_encoder.output_shape(),
                output_dim=policy_config.latent_dim,
                hidden_units=[policy_config.latent_dim, policy_config.latent_dim],
                activation=nn.LeakyReLU(0.2),
                output_activation=None
            )
        )

        self.nets["dynamics"] = MLP(
            input_dim=policy_config.latent_dim + action_dim,
            output_dim=policy_config.latent_dim,
            hidden_units=policy_config.hidden_units,
            activation=nn.LeakyReLU(0.2),
            output_activation=None
        )

        self.nets["obs_decoder"] = ObservationDecoder(
            latent_dim=policy_config.latent_dim,
            config=observation_config,
            keys_to_shapes=keys_to_shapes['obs_shape']
        )

    def get_optimizers_and_schedulers(self, **kwargs):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=5e-5)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1)
    
        return [optimizer], [lr_scheduler]
        
    # def forward(self, batch):
    #     assert batch['actions'].shape[1] == 2, "Time dimension should be 2"
        
    #     batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])
    #     batch['actions'] = self.nets["normalizer"].normalize_by_key(batch['actions'], 'actions')

    #     z = self.nets["obs_encoder"](batch['obs'])
    #     z_and_a = torch.cat([z[:, 0], batch['actions'][:, 0]], dim=-1)
    #     pred_z = self.nets["dynamics"](z_and_a)

    #     output = {
    #         'z': z,
    #         'pred_z': pred_z,
    #     }
    
    #     return output
    
    # def compute_loss(self, batch):
    #     output = self.forward(batch)
        
    #     z = output['z']
    #     pred_z = output['pred_z']

    #     losses = AttrDict()
    #     losses.total = 0
        
    #     # use dynamics loss
    #     losses.dynamics = nn.functional.mse_loss(pred_z, z[:, 1])

    #     # use reconstruction loss
    #     decoded_obs = self.nets["obs_decoder"](z)
    #     losses.reconstruction = 0
    #     for obs_key in decoded_obs.keys():
    #         obs_loss = nn.functional.mse_loss(decoded_obs[obs_key], batch['obs'][obs_key])
    #         losses[obs_key] = obs_loss
    #         losses.reconstruction += obs_loss

    #     losses.total += losses.dynamics + losses.reconstruction
        
    #     return losses
    
    def preprocess_obs(self, obs):
        obs = process_obs_dict(obs, self.config.keys_to_modality) # divides by 255 and changes hwc->chw
        
        obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x[None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x)[None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return obs
    
    def get_embeddings(self, obs):
        self.eval()
        obs = self.nets["normalizer"].normalize(obs)

        z = self.nets["obs_encoder"](obs)
        self.train()
        return z
    
    def reconstruct_obs(self, z):
        self.eval()
        decoded_obs = self.nets["obs_decoder"](z)
        unnormalized_decoded_obs = self.nets["normalizer"].unnormalize(decoded_obs)
        self.train()
        return unnormalized_decoded_obs
    
class ReconstructionInverseDynamicsModel(ReconstructionDynamicsModel):

    def forward(self, batch):
        # assert batch['actions'].shape[1] == 2, "Time dimension should be 2"
        
        # print(batch['obs']['robot0_eef_pos'][0, 0])
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])
        
        batch['actions'] = self.nets["normalizer"].normalize_by_key(batch['actions'], 'actions')
        # print(batch['obs']['robot0_eef_pos'][0, 0])
        z = self.nets["obs_encoder"](batch['obs'])
        # z_and_a = torch.cat([z[:, 1], batch['actions'][:, 0]], dim=-1)
        # pred_z = self.nets["dynamics"](z_and_a)

        output = {
            'z': z,
            # 'pred_z': pred_z,
        }
    
        return output
    
    def compute_loss(self, batch):
        output = self.forward(batch)
        
        z = output['z']
        # pred_z = output['pred_z']

        losses = AttrDict()
        losses.total = 0
        
        # use dynamics loss
        # losses.dynamics = nn.functional.mse_loss(pred_z, z[:, 0])

        # regularization loss
        # losses.regularization = z.pow(2).mean()

        # use reconstruction loss
        decoded_obs = self.nets["obs_decoder"](z)
        losses.reconstruction = 0
        # print(batch['obs']['robot0_eef_pos'][0, 0])
        batch['obs'] = self.nets["normalizer"].normalize(batch['obs'])
        for obs_key in decoded_obs.keys():
            # print(batch['obs'][obs_key][0,0])
            obs_loss = nn.functional.mse_loss(decoded_obs[obs_key], batch['obs'][obs_key])
            losses[obs_key] = obs_loss
            losses.reconstruction += obs_loss
        # print()
        losses.total += losses.reconstruction #+ 0.01*losses.regularization
        
        return losses