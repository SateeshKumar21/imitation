import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation.utils.general_utils import AttrDict
from imitation.algo.base_algo import BaseAlgo
from imitation.utils.lr_scheduler import get_scheduler
from imitation.models.normalizers import DictNormalizer
from imitation.models.obs_nets import ObservationEncoder
from imitation.models.conditional_unet import ConditionalUnet1D
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from einops import reduce

from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.torch_utils import replace_bn_with_gn

from collections import deque

from imitation.utils.flow_utils import FlowTimeSampler
from imitation.models.diffusion_mlp_nets import MLPDiffusionHead


class FlowPolicy(BaseAlgo):

    def __init__(self, config):
        super(FlowPolicy, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']
        
        self.nets = nn.ModuleDict()
        key_to_norm_type = observation_config.obs_keys_to_normalize
        key_to_norm_type['actions'] = config.policy_config.action_normalization_type
        normalizer = DictNormalizer(config.normalization_stats, key_to_norm_type=key_to_norm_type)
        self.nets["normalizer"] = normalizer

        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        obs_encoder = replace_bn_with_gn(obs_encoder)
        self.nets["obs_encoder"] = obs_encoder

        obs_feature_dim = obs_encoder.output_shape()

        # input_dim = action_dim
        # global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        # model = ConditionalUnet1D(
        #     input_dim=input_dim,
        #     global_cond_dim=global_cond_dim,
        #     diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim,
        #     down_dims=policy_config.down_dims,
        #     kernel_size=policy_config.kernel_size,
        #     n_groups=policy_config.n_groups,
        #     cond_predict_scale=policy_config.cond_predict_scale,
        #     pos_embedding_period=policy_config.pos_embedding_period,
        # )

        flat_action_dims = action_dim*policy_config.n_action_steps
        global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        model = MLPDiffusionHead(
            input_dim=flat_action_dims + global_cond_dim + policy_config.diffusion_step_embed_dim,
            output_dim=flat_action_dims,
            diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim
        )

        self.nets["model"] = model

        self.flow_time_sampler = FlowTimeSampler(**policy_config.flow_time_sampler_kwargs)

        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = policy_config.n_action_steps
        self.n_obs_steps = policy_config.n_obs_steps

        # if num_inference_steps is None:
        self.num_inference_steps = policy_config.num_inference_steps
        self.reset()

    def get_optimizers_and_schedulers(self, **kwargs):
        assert 'num_epochs' in kwargs, 'num_epochs is required for DiffusionPolicy optimizer initialization'
        assert 'epoch_every_n_steps' in kwargs, 'epoch_every_n_steps is required for DiffusionPolicy optimizer initialization'

        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-4)
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=kwargs['num_epochs'] * kwargs['epoch_every_n_steps']
        )

        return [optimizer], [lr_scheduler]
        
    def forward(self, batch):
        return None
    
    def psi_t(self, x: torch.FloatTensor, x1: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_time_sampler.flow_sig_min) * t) * x + t * x1
    
    def compute_loss(self, batch):
        actions = batch['actions'][:, -1]
        actions = self.nets["normalizer"].normalize_by_key(actions, 'actions')
        B = actions.shape[0]

        # obs as global cond
        required_obs = recursive_dict_list_tuple_apply(batch['obs'], {torch.Tensor: lambda x: x[:, :self.n_obs_steps, ...].clone()})
        normalized_obs = self.nets["normalizer"].normalize(required_obs)

        obs_features = self.nets["obs_encoder"](normalized_obs)
        global_cond = obs_features.reshape(B, -1)

        noise = torch.randn_like(actions, device=actions.device)
        timesteps = self.flow_time_sampler.sample_fm_time(B).to(actions.device)
        psi_t = self.psi_t(noise, actions, timesteps)

        v_psi = self.nets["model"](psi_t, timesteps, global_cond=global_cond)
        d_psi = actions - (1 - self.flow_time_sampler.flow_sig_min) * noise

        losses = AttrDict()
        loss = F.mse_loss(v_psi, d_psi, reduction='none')
        
        loss = reduce(loss, 'b t a -> b', 'mean')
        
        losses.mse = loss.mean()
        losses.total = losses.mse
        return losses
    
    def post_step_update(self):
        pass

    def preprocess_obs(self, obs, batched=False):
        obs = process_obs_dict(obs, self.config.keys_to_modality) # divides by 255 and changes hwc->chw

        if not batched:
            obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x[None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x)[None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        else:
            obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x.float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x).float().to(self.device),
                    type(None): lambda x: x,
                }
            )

        return obs

    @torch.no_grad()
    def get_action(self, obs, batched=False):
        '''
            obs expected to have no time dimension
        '''
        
        self.eval()

        model = self.nets
        obs = self.preprocess_obs(obs, batched=batched)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)
        
        # make sure obs_queue is full
        self.obs_queue.append(obs_features)
        while len(self.obs_queue) < self.n_obs_steps:
            self.obs_queue.append(obs_features.clone())
        
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.train()
            return action
        
        global_cond = torch.cat(list(self.obs_queue), dim=1)
        pred_action = torch.randn((global_cond.shape[0], self.n_action_steps, self.action_dim), device=obs_features.device)
        
        delta_t = 1.0 / self.num_inference_steps
        timestep = torch.zeros(global_cond.shape[0], device=obs_features.device)
        for _ in range(self.num_inference_steps):
            model_output = model["model"](pred_action, timestep, global_cond=global_cond)

            pred_action += delta_t * model_output
            timestep += delta_t
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.permute(1, 0, 2)
        self.action_queue.extend(pred_action.cpu().numpy())

        self.train()
        action = self.action_queue.popleft()
        return action

    def to(self, device):
        super().to(device)

    def reset(self):
        self.obs_queue = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque(maxlen=self.n_action_steps)

    @torch.no_grad()
    def get_output(self, obs):
        '''
            obs expected to have no time dimension
        '''
        
        self.eval()

        model = self.nets
    
        obs = self.preprocess_obs(obs, batched=True)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)

        B = obs_features.shape[0]
        global_cond = obs_features.reshape(B, -1)
        pred_action = torch.randn((B, self.n_action_steps, self.action_dim), device=obs_features.device)
    
        delta_t = 1.0 / self.num_inference_steps
        timestep = torch.zeros(B, device=obs_features.device)
        for _ in range(self.num_inference_steps):
            model_output = model["model"](pred_action, timestep, global_cond=global_cond)

            pred_action += delta_t * model_output
            timestep += delta_t
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')

        self.train()
        return pred_action.cpu().numpy()