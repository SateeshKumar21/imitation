import torch
import torch.nn as nn
from imitation.models.positional_embedding import SinusoidalPosEmb
from imitation.models.base_nets import MLP


class MLPDiffusionHead(nn.Module):
    def __init__(self, input_dim, output_dim, diffusion_step_embed_dim):
        super(MLPDiffusionHead, self).__init__()
        self.net = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=[512, 512, 512],
            normalization=None,
            activation=nn.ReLU(),
            output_activation=None
        )

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

    def forward(self, sample, timesteps, global_cond):
        B, T, _ = sample.shape
        sample = sample.view(B, -1)

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        timesteps = timesteps.expand(sample.shape[0])

        time_emb = self.diffusion_step_encoder(timesteps.float())
        global_cond = torch.cat([global_cond, time_emb], dim=-1)

        sample = torch.cat([sample, global_cond], dim=-1)

        pred = self.net(sample)
        pred = pred.view(B, T, -1)
        return pred