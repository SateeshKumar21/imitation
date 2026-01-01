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
from imitation.models.ema_model import EMAModel
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from einops import reduce

from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.torch_utils import replace_bn_with_gn

from collections import deque
from imitation.models.diffusion_mlp_nets import MLPDiffusionHead


class DiffusionPolicy(BaseAlgo):

    def __init__(self, config):
        super(DiffusionPolicy, self).__init__()
        
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

        input_dim = action_dim
        global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim,
            down_dims=policy_config.down_dims,
            kernel_size=policy_config.kernel_size,
            n_groups=policy_config.n_groups,
            cond_predict_scale=policy_config.cond_predict_scale
        )
        
        # flat_action_dims = action_dim*policy_config.n_action_steps
        # global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        # model = MLPDiffusionHead(
        #     input_dim=flat_action_dims + global_cond_dim + policy_config.diffusion_step_embed_dim,
        #     output_dim=flat_action_dims,
        #     diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim
        # )

        self.nets["model"] = model

        self.noise_scheduler = policy_config.noise_schedular_class(
            **policy_config.noise_schedular_kwargs
        )

        self.ema = None
        if policy_config.use_ema:
            self.ema = EMAModel(model=copy.deepcopy(self.nets), power=policy_config.ema_power)

        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = policy_config.n_action_steps
        self.n_obs_steps = policy_config.n_obs_steps
        self.input_pertub = policy_config.input_pertub
        self.use_ema = policy_config.use_ema

        # if num_inference_steps is None:
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
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
    
    def compute_loss(self, batch):
        #import pdb; pdb.set_trace()
        actions = batch['actions'][:, -1]
        actions = self.nets["normalizer"].normalize_by_key(actions, 'actions')
        B = actions.shape[0]

        # obs as global cond
        required_obs = recursive_dict_list_tuple_apply(batch['obs'], {torch.Tensor: lambda x: x[:, :self.n_obs_steps, ...].clone()})
        normalized_obs = self.nets["normalizer"].normalize(required_obs)

        obs_features = self.nets["obs_encoder"](normalized_obs)
        global_cond = obs_features.reshape(B, -1)

        noise = torch.randn(actions.shape, device=actions.device)
        noise_new = noise + self.input_pertub * torch.randn(actions.shape, device=actions.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=actions.device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, noise_new, timesteps)

        pred = self.nets["model"](noisy_actions, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = actions
        else:
            raise NotImplementedError

        losses = AttrDict()
        loss = F.mse_loss(pred, target, reduction='none')
        
        # loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b t a -> b', 'mean')
        
        losses.mse = loss.mean()
        losses.total = losses.mse
        return losses
    
    def post_step_update(self):
        if self.use_ema:
            self.ema.step(self.nets)

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
        if self.use_ema:
            model = self.ema.averaged_model
        #import pdb; pdb.set_trace()
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
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = model["model"](pred_action, t, global_cond=global_cond)

            pred_action = self.noise_scheduler.step(
                model_output=model_output, 
                timestep=t,
                sample=pred_action).prev_sample
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.permute(1, 0, 2)  # (B, action_dim, n_action_steps)
        self.action_queue.extend(pred_action.cpu().numpy())

        self.train()
        action = self.action_queue.popleft()
        return action

    @torch.no_grad()
    def get_action_debug(self, obs, batched=False):
        '''
            obs expected to have no time dimension
        '''
        #import pdb; pdb.set_trace()
        self.eval()

        model = self.nets
        if self.use_ema:
            model = self.ema.averaged_model
       
       
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)
        
        # make sure obs_queue is full
        # self.obs_queue.append(obs_features)
        # while len(self.obs_queue) < self.n_obs_steps:
        #     self.obs_queue.append(obs_features.clone())
        
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.train()
            return action
        self.obs_queue = obs_features
        #global_cond = torch.cat(list(self.obs_queue), dim=1)
        global_cond = self.obs_queue.reshape(1, -1)
        pred_action = torch.randn((global_cond.shape[0], self.n_action_steps, self.action_dim), device=obs_features.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = model["model"](pred_action, t, global_cond=global_cond)

            pred_action = self.noise_scheduler.step(
                model_output=model_output, 
                timestep=t,
                sample=pred_action).prev_sample
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.permute(1, 0, 2)  # (B, action_dim, n_action_steps)
        self.action_queue.extend(pred_action.cpu().numpy())

        self.train()
        action = self.action_queue.popleft()
        return action


    def to(self, device):
        super().to(device)
        if self.use_ema:
            self.ema.to(device)

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
        if self.use_ema:
            model = self.ema.averaged_model
    
        obs = self.preprocess_obs(obs, batched=True)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)

        global_cond = obs_features.reshape(obs_features.shape[0], -1)
        pred_action = torch.randn((global_cond.shape[0], self.n_action_steps, self.action_dim), device=obs_features.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = model["model"](pred_action, t, global_cond=global_cond)

            pred_action = self.noise_scheduler.step(
                model_output=model_output, 
                timestep=t,
                sample=pred_action).prev_sample
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')

        self.train()
        return pred_action.cpu().numpy()


    