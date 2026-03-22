from imitation.algo.diffusion_policy import DiffusionPolicy
import torch
from imitation.envs.real_env import TiagoGym, WindowBufferWrapper
import numpy as np

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--demo_index', type=int, default=11)
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of steps to run. Defaults to full demo length.')
    parser.add_argument('--window_size', type=int, default=None,
                        help='If set, wraps env with WindowBufferWrapper.')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Resize images to this size. Skip if None or matches data.')
    args = parser.parse_args()

    env = TiagoGym(data_dir=args.data_dir, demo_index=args.demo_index)

    if args.window_size is not None:
        env = WindowBufferWrapper(env, window_size=args.window_size)

    env_obs, _ = env.reset()

    model = DiffusionPolicy.load_weights(args.ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_steps = args.num_steps if args.num_steps is not None else env.num_samples
    demo_key = f'data/demo_{args.demo_index}'

    for i in range(num_steps):
        obs, _, done, truncated, _ = env.step(env.action_space.sample())

        if args.image_size is not None:
            import cv2
            image = obs['tiago_head_image'].astype(np.float32)
            obs['tiago_head_image'] = cv2.resize(image, (args.image_size, args.image_size))

        action = model.get_action(obs, batched=False)

        gt_action = env.hdf5_object[demo_key]['actions'][env.curr_index]
        print(f"Step {i}: predicted={action}")
        print(f"         GT action={gt_action}")

        if done or truncated:
            print(f"Episode ended at step {i}")
            break

    env.close()
