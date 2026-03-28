from imitation.utils.general_utils import AttrDict
from imitation.algo.diffusion_policy import DiffusionPolicy
from imitation.models.image_nets import ResNet18, SpatialSoftmax
from imitation.models.obs_nets import VisionCore, LowDimCore
from imitation.data.dataset import SequenceDataset
import torch.nn as nn
import diffusers
from collections import OrderedDict
from imitation.models.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug
window_size = 5
action_horizon = 8

train_config = AttrDict(
    output_dir="experiments/",
    batch_size=32,
    num_epochs=400,
    epoch_every_n_steps=100,
    log_every_n_epochs=1,
    val_every_n_epochs=50,
    save_every_n_epochs=50,
    eval_every_n_epochs=25,
    seed=1
)

data_config = AttrDict(
    data=[
        "/home/ec2-user/augmented_il_data_jan_5/augmented_il_data_jan_5_v1/augmented_v1.h5"
        # "/mnt/hdd2/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
        # "/mnt/hdd2/libero/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",

    ],
    dataset_class=SequenceDataset,
    dataset_kwargs=dict(
        dataset_keys=['actions'],
        window_size=window_size,
        action_horizon=action_horizon,
    ),
    num_workers=2
)

policy_config = AttrDict(
    policy_class=DiffusionPolicy,
    n_obs_steps=window_size,
    n_action_steps=action_horizon,
    diffusion_step_embed_dim=128,
    down_dims=[256,512,1024],
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
    input_pertub=0.1,

    use_ema=False,
    ema_power=0.75,

    noise_schedular_class=diffusers.DDIMScheduler,
    noise_schedular_kwargs=dict(
        num_train_timesteps=50,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='sample'
    ),
    # noise_schedular_class=diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    # noise_schedular_kwargs=dict(
    #     num_train_timesteps=100,
    #     beta_start=1e-4,
    #     beta_end=2e-2,
    #     beta_schedule="squaredcos_cap_v2",
    #     clip_sample=False,
    #     variance_type='fixed_small',
    #     prediction_type='epsilon'
    # ),
    action_normalization_type='gaussian', #'bounds'
)

observation_config = AttrDict(
    obs = OrderedDict(
        low_dim = [
           
        ],
        rgb = [
            "tiago_head_image",
        ],
        depth = [],
    ),
    obs_keys_to_normalize = {
 
    },
    encoder = AttrDict(
        low_dim = AttrDict(
            core_class=None,#LowDimCore,
            core_kwargs=dict(
                output_dim=16,
                hidden_units=[16],
                activation=nn.LeakyReLU(0.2),
                output_activation=nn.LeakyReLU(0.2),
            ),
        ),
        rgb = AttrDict(
            core_class=VisionCore,
            core_kwargs=dict(
                backbone_class=ResNet18,
                backbone_kwargs=None,
                feature_dim=64,
                pool_class=SpatialSoftmax,
                pool_kwargs=dict(
                    num_kp=32,
                    learnable_temperature=False,
                    temperature=1.0,
                    noise_std=0.0,
                ),
            ),
        ),
         augmentation=[
                AttrDict(
                    aug_class=TranslationAug,
                    aug_kwargs=dict(
                        translation=24
                    )
                ),
                AttrDict(
                    aug_class=BatchWiseImgColorJitterAug,
                    aug_kwargs=dict(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.3,
                        epsilon=0.1
                    )
                )
            ],
        depth = AttrDict(
            core_class=None,
            core_kwargs=dict(),
        ),
    )
)

evaluator_config = None
# from imitation.evaluators.libero_evaluator import LiberoEvaluator
# evaluator_config1 = AttrDict(
#     evaluator=LiberoEvaluator,
#     task_id=5,  # Change this to the desired task ID
#     num_envs=25,
#     total_evals=25,
#     num_steps=500,
#     save_vid=False,
#     save_path='/home/shivin/compi/experiments/videos',
# )

# from imitation.evaluators.vis_action_distribution import VisActionDistribution
# evaluator_config2 = AttrDict(
#     evaluator=VisActionDistribution,
#     n_points=1000, 
# )

# from imitation.evaluators.base_evaluator import MultiEvals
# evaluator_config = AttrDict(
#     evaluator=MultiEvals,
#     evaluator_configs=[evaluator_config1, evaluator_config2],
# )

# from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator
# from l2l.config.env.robosuite.multi_stage import env_config
# evaluator_config = AttrDict(
#     evaluator=RobosuiteEvaluator,
#     env_config = env_config,
#     n_rollouts = 10,
#     max_steps = 100,
# )

config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=evaluator_config,
)