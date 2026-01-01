import numpy as np
from imitation.algo.bc_ce import BC_CrossEntropy
from imitation.algo.bc_rnn import BC_RNN
from imitation.algo.diffusion_policy import DiffusionPolicy
from imitation.algo.bc_transformer import BCTransformer
import torch
import cv2
from imitation.utils.vis_utils import get_saliency_maps, write_video

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    # Load the module
    # model = BC_CrossEntropy.load_weights(args.ckpt)
    model = BCTransformer.load_weights(args.ckpt)
    # model = DiffusionPolicy.load_weights(args.ckpt)
    # model = BC_RNN.load_weights(args.ckpt)

    import robosuite as suite
    from l2l.config.env.robosuite.multi_stage import env_config
    # from l2l.config.env.robosuite.skill_color_pnp import env_config
    # from l2l.config.env.robosuite.skill_camera_color_pnp import env_config
    # from l2l.config.env.robosuite.color_lift import env_config
    # from l2l.config.env.robosuite.camera_color_pnp_imitation import env_config
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
    n_evals = 0
    n_success = 0
    for i in range(10000):
        action = model.get_action(obs)#np.random.uniform(low, high)

        # saliency_maps = list(get_saliency_maps(model, obs).values())
        # img = np.concatenate(saliency_maps, axis=1)
        # img = np.concatenate([img, obs['sideview_image'].astype(np.uint8)], axis=1)
        img = np.concatenate([obs['agentview_image']/255, obs['activeview_image']/255], axis=1)
        # img = obs['activeview_image']/255

        video.append(img)
        for _ in range(5):
            cv2.imshow('img', img)
            cv2.waitKey(5)

        obs, reward, done, _ = env.step(action)
        
        # for k in obs:
        #     if 'image' in k:
        #         obs[k] = obs[k].astype(np.uint8)

        steps += 1

        if done or steps%150==0:
            obs = env.reset()
            steps = 0
            model.reset()

            n_evals += 1
            if done:
                n_success += 1


            # random_init = np.random.choice(object_pos)
            # env.sim.data.set_joint_qpos(env.cubeA.joints[0], np.concatenate([np.array(random_init['cube_red']['pos']), np.array(random_init['cube_red']['quat'])]))
            # env.sim.data.set_joint_qpos(env.cubeB.joints[0], np.concatenate([np.array(random_init['cube_green']['pos']), np.array(random_init['cube_green']['quat'])]))
            
            # obs, reward, done, _ = env.step(np.zeros_like(low))
            if n_evals > 10:
                break

    print(f"Success rate: {n_success/n_evals}")
    # write_video(video, "skill_actiview_pnp_with_saliency.mp4", fps=10)