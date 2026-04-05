import os
from tqdm import tqdm
import torch
import torch.cuda.amp as amp
import numpy as np
from torch.utils.data import DataLoader
from imitation.utils.general_utils import AttrDict
from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.log_utils import log_value_in_dict, WandBLogger
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from imitation.utils.file_utils import get_all_obs_keys_from_config, get_shape_metadata_from_dataset, get_obs_key_to_modality_from_config
from importlib.machinery import SourceFileLoader

DEVICE = 'cuda'
torch.set_float32_matmul_precision('high')  # enable TF32 on Ampere/Ada GPUs

LOG = True
WANDB_PROJECT_NAME = '3dmoma'
WANDB_ENTITY_NAME = 'sakumar'

class Trainer:

    def __init__(self, config_path, exp_name, resume_path=None):
        self.config_path = config_path
        self.exp_name = exp_name
        self.load_config(config_path)

        self.create_log()
        self.setup_data()
        self.setup_model()

        self.start_epoch = 0
        if resume_path is not None:
            self.load_checkpoint(resume_path)

        self.evaluator = None
        if self.evaluator_config:
            self.evaluator = self.evaluator_config.evaluator(eval_config=self.evaluator_config, trainer=self)

    def load_checkpoint(self, path):
        print(f"==> Resuming from checkpoint: {path}")
        _, state = torch.load(path, weights_only=False)
        self.model.load_state_dict(state)
        # parse epoch from filename, e.g. weights_ep100.pth
        basename = os.path.basename(path)
        try:
            self.start_epoch = int(basename.split('ep')[1].split('.')[0])
            print(f"==> Resuming from epoch {self.start_epoch}")
        except (IndexError, ValueError):
            self.start_epoch = 0

    def train(self, n_epochs):
        self.model.train()
        # Visualize at start of training
        if self.log_path is not None:
            self.visualize_training_images(0)
        for epoch in range(self.start_epoch, n_epochs):
            epoch_info = self.train_epoch(epoch)
            self.model.post_epoch_update()
            
            if epoch % self.train_config.log_every_n_epochs == 0 :
                losses = epoch_info.losses
                if self.logger is not None:
                    self.logger.log_scalar_dict(losses, step=epoch, phase='train/losses')
                    self.logger.log_scalar(epoch_info.gradient_norm, 'gradients/mean_norm', epoch, 'train')
                    self.logger.log_scalar(epoch_info.weight_norm, 'weights/mean_norm', epoch, 'train')

                print(f'\nepoch {epoch}')
                print('Losses')
                for loss in losses.keys():
                    print(f'\t{loss}: {losses[loss]}', end='\n\n') 

            if self.train_config.val_every_n_epochs > 0 and (epoch+1) % self.train_config.val_every_n_epochs == 0:
                val_info = self.validate()
                if self.logger is not None:
                    self.logger.log_scalar_dict(val_info, step=epoch, phase='val')
                
            if (self.evaluator is not None) and self.train_config.eval_every_n_epochs > 0 and (epoch+1) % self.train_config.eval_every_n_epochs == 0:
                eval_info = self.evaluate()
                if self.logger is not None:
                    self.logger.log_multi_modal_dict(eval_info, step=epoch, phase='eval')

            if self.train_config.save_every_n_epochs > 0 and (epoch+1) % self.train_config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch+1)
                if self.log_path is not None:
                    self.visualize_training_images(epoch+1)

    def train_epoch(self, epoch):
        
        data_loader_iter = iter(self.train_loader)
        epoch_info = AttrDict(
            losses=AttrDict(),
            gradient_norm=0,
            weight_norm=0
        )
        for batch_idx in tqdm(range(self.train_config.epoch_every_n_steps)):

            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(self.train_loader)
                batch = next(data_loader_iter)
            #import pdb; pdb.set_trace()
            batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE, non_blocking=True).float()})
            batch['obs'] = process_obs_dict(batch['obs'], self.obs_key_to_modality)
            
            # batchnorm doesn't work with batch size 1 
            if batch['actions'].shape[0] == 1:
                continue
            
            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                losses = self.model.compute_loss(batch)

            if self.scaler is not None:
                self.scaler.scale(losses.total).backward()
            else:
                losses.total.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            for i in range(len(self.optimizers)):
                if self.scaler is not None:
                    self.scaler.step(self.optimizers[i])
                else:
                    self.optimizers[i].step()

                if self.lr_schedulers[i] is not None:
                    self.lr_schedulers[i].step()

            if self.scaler is not None:
                self.scaler.update()

            self.model.post_step_update()

            # logging
            for k in losses:
                if k not in epoch_info.losses:
                    epoch_info.losses[k] = 0
                epoch_info.losses[k] += float(losses[k].item())/self.train_config.epoch_every_n_steps

        # compute norms once at end of epoch (grads from last step are still valid)
        epoch_info.gradient_norm = torch.mean(torch.stack([torch.norm(p.grad.data) for p in self.model.parameters() if p.grad is not None]))
        epoch_info.weight_norm = torch.mean(torch.stack([torch.norm(p.data) for p in self.model.parameters() if p.grad is not None]))

        return epoch_info

    def validate(self):
        print("\nValidating policy...")

        #import pdb; pdb.set_trace()
        self.model.eval()
        with torch.no_grad():
            val_loss = AttrDict()

            for batch in tqdm(self.val_loader):
                batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE, non_blocking=True).float()})
                batch['obs'] = process_obs_dict(batch['obs'], self.obs_key_to_modality)

                losses = self.model.compute_loss(batch)

            # logging
            for k in losses:
                if k not in val_loss:
                    val_loss[k] = 0
                val_loss[k] += float(losses[k].item())/len(self.val_loader)

            print('#'*20, '\nValidation Losses')
            for k in val_loss.keys():
                print(f'\t{k}: {val_loss[k]}', end='\n\n') 
            print('#'*20)

        self.model.train()
        return val_loss

    def evaluate(self):
        print("\nEvaluating policy...")
        return self.evaluator.evaluate(self.model)

    def create_log(self):
        self.logger = None
        self.log_path = None

        if LOG:
            self.log_path = os.path.join(self.train_config.output_dir, self.exp_name)
            # TODO: create weights directory and fix the model save path
            os.makedirs(self.log_path, exist_ok=True)

            self.logger = WandBLogger(self.exp_name, WANDB_PROJECT_NAME, WANDB_ENTITY_NAME, self.log_path, self.conf)

    def save_checkpoint(self, epoch):
        if self.log_path is not None:
            print(f"==> Saving checkpoint at epoch {epoch}")
            self.model.save_weights(epoch, self.log_path)

    def visualize_training_images(self, epoch):
        """Save a grid of training images (original vs augmented) to disk."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vis_dir = os.path.join(self.log_path, 'training_vis')
        os.makedirs(vis_dir, exist_ok=True)

        rgb_keys = [k for k, v in self.obs_key_to_modality.items() if v == 'rgb']
        if not rgb_keys:
            return

        # Grab one batch
        batch = next(iter(self.train_loader))
        batch['obs'] = process_obs_dict(batch['obs'], self.obs_key_to_modality)
        batch = recursive_dict_list_tuple_apply(batch, {torch.Tensor: lambda x: x.to(DEVICE).float()})

        obs_encoder = self.model.nets["obs_encoder"]
        n_show = min(4, batch['obs'][rgb_keys[0]].shape[0])

        for rgb_key in rgb_keys:
            images = batch['obs'][rgb_key][:n_show]  # (N, T, C, H, W)
            T = images.shape[1]

            # Apply augmentation
            obs_encoder.train()
            with torch.no_grad():
                aug_images = obs_encoder.apply_augmentation(rgb_key, images.clone())

            fig, axes = plt.subplots(n_show * 2, T, figsize=(3 * T, 2.5 * n_show * 2))
            if T == 1:
                axes = axes[:, None]

            for s in range(n_show):
                for t in range(T):
                    orig = images[s, t].detach().cpu().numpy().transpose(1, 2, 0)
                    orig = (orig * 255).clip(0, 255).astype(np.uint8)
                    axes[s * 2, t].imshow(orig)
                    axes[s * 2, t].set_title(f's{s} t{t} original', fontsize=7)
                    axes[s * 2, t].axis('off')

                    aug = aug_images[s, t].detach().cpu().numpy().transpose(1, 2, 0)
                    aug = (aug * 255).clip(0, 255).astype(np.uint8)
                    axes[s * 2 + 1, t].imshow(aug)
                    axes[s * 2 + 1, t].set_title(f's{s} t{t} augmented', fontsize=7)
                    axes[s * 2 + 1, t].axis('off')

            plt.suptitle(f'Epoch {epoch} | {rgb_key} | original vs augmented', fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_path = os.path.join(vis_dir, f'epoch_{epoch}_{rgb_key}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved training vis: {out_path}")

    def load_config(self, config_path):
        self.conf = SourceFileLoader('conf', config_path).load_module().config
        
        self.train_config = self.conf.train_config
        self.data_config = self.conf.data_config
        self.policy_config = self.conf.policy_config
        self.observation_config = self.conf.observation_config

        self.obs_keys = get_all_obs_keys_from_config(self.observation_config)
        self.obs_key_to_modality = get_obs_key_to_modality_from_config(self.observation_config)
        self.shape_meta = get_shape_metadata_from_dataset(self.data_config.data[0], all_obs_keys=self.obs_keys, obs_key_to_modality=self.obs_key_to_modality)
        
        self.evaluator_config = self.conf.evaluator_config if 'evaluator_config' in self.conf else None

    def setup_data(self):
        self.train_dataset = self.data_config.dataset_class(
                                data_paths=self.data_config.data,
                                obs_keys_to_modality=self.obs_key_to_modality,
                                obs_keys_to_normalize=self.observation_config.obs_keys_to_normalize,
                                split='train',
                                **self.data_config.dataset_kwargs)
        self.normalization_stats = self.train_dataset.get_normalization_stats()

        if self.train_config.val_every_n_epochs > 0:
            self.val_dataset = self.data_config.dataset_class(
                                data_paths=self.data_config.data,
                                obs_keys_to_modality=self.obs_key_to_modality,
                                obs_keys_to_normalize=self.observation_config.obs_keys_to_normalize,
                                split='val',
                                **self.data_config.dataset_kwargs)
            
        self.setup_dataloader()
    
    def setup_dataloader(self):
        self.train_loader = DataLoader(
                                self.train_dataset,
                                batch_size=self.train_config.batch_size,
                                shuffle=True,
                                num_workers=self.data_config.num_workers,
                                pin_memory=True,
                                persistent_workers=self.data_config.num_workers > 0,
                                prefetch_factor=2 if self.data_config.num_workers > 0 else None)

        if self.train_config.val_every_n_epochs > 0:
            self.val_loader = DataLoader(
                                self.val_dataset,
                                batch_size=self.train_config.batch_size,
                                shuffle=True,
                                num_workers=self.data_config.num_workers,
                                pin_memory=True,
                                persistent_workers=self.data_config.num_workers > 0,
                                prefetch_factor=2 if self.data_config.num_workers > 0 else None)

    def setup_model(self):
        model_config = AttrDict(
            policy_config=self.policy_config,
            observation_config=self.observation_config,
            keys_to_shapes=self.shape_meta,
            keys_to_modality=self.obs_key_to_modality,
            normalization_stats=self.normalization_stats
        )
        self.model = self.policy_config.policy_class(model_config)
        self.model.to(DEVICE)
        self.model.train()

        self.optimizers, self.lr_schedulers = self.model.get_optimizers_and_schedulers(
            num_epochs=self.train_config.num_epochs,
            epoch_every_n_steps=self.train_config.epoch_every_n_steps
        )
        assert len(self.optimizers) == len(self.lr_schedulers), 'Number of optimizers and learning rate schedulers should be same'

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # bfloat16 doesn't need GradScaler
        else:
            self.amp_dtype = torch.float16
            self.scaler = amp.GradScaler()

        if hasattr(torch, 'compile'):
            print("==> Compiling compute_loss with torch.compile...")
            self.model.compute_loss = torch.compile(self.model.compute_loss, mode='reduce-overhead')

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--exp_name", type=str, help="unique name of the current run(include details of the architecture. eg. SimpleC2R_64x3_relu_run1)")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint .pth file to resume training from")
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config, exp_name=args.exp_name, resume_path=args.resume)
    trainer.train(n_epochs=trainer.train_config.num_epochs)
