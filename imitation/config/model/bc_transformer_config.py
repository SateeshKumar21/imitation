from imitation.utils.general_utils import AttrDict
from imitation.algo.bc_transformer import BCTransformer
from imitation.models.distribution_nets import MDN, Gaussian
from imitation.models.image_nets import ResNet18, ResNet34, SpatialSoftmax, R3M, VisionTransformer
from imitation.models.obs_nets import VisionCore, LowDimCore
from imitation.data.dataset_old import SequenceDatasetMultiFile
import torch.nn as nn
from collections import OrderedDict

train_config = AttrDict(
    output_dir="/home/shivin/compi/experiments/",
    batch_size=16,
    num_epochs=1000,
    epoch_every_n_steps=500,
    log_every_n_epochs=1,
    val_every_n_epochs=10,
    save_every_n_epochs=25,
    eval_every_n_epochs=25,
    seed=1
)

data_config = AttrDict(
    data=["/mnt/hdd2/libero/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5"],
    dataset_class=SequenceDatasetMultiFile,
    dataset_kwargs=dict(
        seq_length=10,
        pad_seq_length=True,
        frame_stack=1,
        pad_frame_stack=True,
        dataset_keys=['actions'],
        hdf5_cache_mode='low_dim',
        hdf5_use_swmr=True
    ),
    num_workers=2
)

# BC-Transformer GMM
policy_config = AttrDict(
    policy_class=BCTransformer,
    token_dim=64,
    transformer_num_layers=4,
    transformer_num_heads=6,
    transformer_head_output_size=64,
    transformer_mlp_hidden_size=256,
    transformer_dropout=0.1,
    horizon=10,
 
    action_head=MDN,
    action_head_kwargs=dict(
        num_gaussians=5
    ),
    action_normalization_type='gaussian',
)

observation_config = AttrDict(
    obs = OrderedDict(
        low_dim = [
            "ee_pos",
            "ee_ori",
            "gripper_states",
        ],
        rgb = [
            "agentview_rgb",
            "eye_in_hand_rgb",
        ],
        depth = [],
    ),
    obs_keys_to_normalize = {
        "ee_pos": 'gaussian',
        "ee_ori": 'gaussian',
        "gripper_states": 'gaussian',
    },
    encoder = AttrDict(
        low_dim = AttrDict(
            core_class=LowDimCore,
            core_kwargs=dict(
                feature_dim=policy_config.token_dim,
                hidden_units=[],
                activation=nn.LeakyReLU(0.2),
                output_activation=None,
            ),
        ),
        rgb = AttrDict(
            core_class=VisionCore,
            core_kwargs=dict(
                backbone_class=ResNet18,
                backbone_kwargs=None,
                feature_dim=policy_config.token_dim,
                pool_class=SpatialSoftmax,
                pool_kwargs=dict(
                    num_kp=32,
                    learnable_temperature=False,
                    temperature=1.0,
                    noise_std=0.0,
                ),
            ),
        ),
        depth = AttrDict(
            core_class=None,
            core_kwargs=dict(),
        ),
    )
)

evaluator_config = None
# from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator
# from l2l.config.env.robosuite.color_lift import env_config
# evaluator_config = AttrDict(
#     evaluator=RobosuiteEvaluator,
#     env_config = env_config,
#     n_rollouts = 10,
#     max_steps = 60,
# )


config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=evaluator_config,
)
