import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import torch

class VisActionDistribution:

    def __init__(self, eval_config, trainer, *args, **kwargs):
        self.config = eval_config
        self.trainer = trainer
        self.n_points = eval_config.get('n_points', 1000)

        # self.log_path = os.path.join(trainer.log_path, 'images')

    def evaluate(self, policy):
        true_actions =  []
        pred_actions = []
        all_timesteps = []

        seen_timesteps = 0

        print("\nEvaluating action distribution...")
        data_loader_iter = iter(self.trainer.val_loader)
        for batch in tqdm(data_loader_iter):
            
            idxs = batch['idx'][:, None] + np.arange(0, policy.n_action_steps)[None]
            idxs = idxs.reshape(-1)
            all_timesteps.append(idxs)

            true_actions.append(batch['actions'][:, -1].reshape(-1, policy.action_dim))

            output = policy.get_output(batch['obs'])
            pred_actions.append(output.reshape(-1, policy.action_dim))

            if len(true_actions) >= self.n_points:
                break

            seen_timesteps += batch['actions'][:, -1].reshape(-1, policy.action_dim).shape[0]
            if seen_timesteps >= self.n_points: 
                break            

        true_actions = np.concatenate(true_actions, axis=0)
        pred_actions = np.concatenate(pred_actions, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)

        fig, axes = plt.subplots(policy.action_dim, 1, figsize=(10, 3 * policy.action_dim))
        for i in range(policy.action_dim):
            ax = axes[i]
            ax.scatter(all_timesteps, true_actions[:, i], label='True Actions', alpha=0.5)
            ax.scatter(all_timesteps, pred_actions[:, i], label='Predicted Actions', alpha=0.5)
            ax.set_title(f'Action Dimension {i}')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Action Value')
            # ax.set_ylimit(-1, 1)  # Assuming actions are normalized between -1 and 1
            ax.legend()

        # plt.savefig('tmp.png')
        return {'action_distribution': fig, 'key_to_modality': {'action_distribution': 'plot'}}
    