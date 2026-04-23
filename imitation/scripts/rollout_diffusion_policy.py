import os
import re
import time
import cv2
import copy
import queue
import threading
import rospy
import numpy as np
import imageio
import h5py
import sys
TELEMOMA_PATH = "/telemoma/telmoma-sateesh/telemoma"
IK_PATH = "/telemoma/tracikpy"
sys.path.insert(0, str(TELEMOMA_PATH))
sys.path.insert(0, str(IK_PATH))

from telemoma.robot_interface.tiago.tiago_gym import TiagoGym
from telemoma.robot_interface.tiago.head import LookAtFixedPoint

rospy.init_node('tiago_rollout_policy')

from telemoma.human_interface.teleop_policy import TeleopPolicy
from imitation.algo.diffusion_policy import DiffusionPolicy
from telemoma.configs.zed_vr import teleop_config
#from gymnasium.wrappers import FrameStackObservation
import torch


def _crop_and_resize(img, size=224):
    """Center-square-crop then resize to (size, size).

    Works for both 1920×1080 ZED frames and smaller ROS camera frames.
    """
    h, w = img.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    cropped = img[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2]
    return cv2.resize(cropped.astype(np.float32), (size, size))


def _next_rollout_vid_name(out_dir):
    """Pick the next rollout_{n} name based on existing rollout_{n}.mp4 files."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = re.compile(r'^rollout_(\d+)(?:_full)?\.mp4$')
    indices = []
    for fname in os.listdir(out_dir):
        m = pattern.match(fname)
        if m:
            indices.append(int(m.group(1)))
    n = max(indices) + 1 if indices else 0
    return f'rollout_{n}'


def _make_save_path(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = os.listdir(save_dir)
    indices = []
    for fname in existing:
        if fname.startswith('demo_') and fname.endswith('.h5'):
            try:
                indices.append(int(fname[len('demo_'):-len('.h5')]))
            except ValueError:
                pass
    demo_id = max(indices) + 1 if indices else 0
    return os.path.join(save_dir, f'demo_{demo_id}.h5')


SINGLE_HAND=True
def rollout_policy(model_ckpt, save_vid=False, vid_name=None, out_dir="./", save_dir=None, execute_horizon=None):

    # load policy
    os.makedirs(out_dir, exist_ok=True)
    if vid_name is None:
        vid_name = _next_rollout_vid_name(out_dir)
        print(f'Auto-selected vid_name: {vid_name}')
    save_path = _make_save_path(save_dir) if save_dir is not None else None
    model = DiffusionPolicy.load_weights(model_ckpt)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    env = TiagoGym(
            frequency=10,
            head_policy=None,
            base_enabled=teleop_config.base_controller is not None,
            right_arm_enabled=teleop_config.arm_right_controller is not None,
            left_arm_enabled=True,
            right_gripper_type=None,
            left_gripper_type='pal',
            camera_type=getattr(teleop_config, 'camera', 'ros'),
            )

    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    obs, _ = env.reset()
    model.reset()

    # --- HDF5 writer (mirrors collect_data_v2.py) ---
    obs_keys_to_save = list(obs.keys())
    WRITE_BATCH = 16
    write_queue = queue.Queue()

    def _writer_thread():
        h5_f = None
        obs_ds = {}
        action_ds = reward_ds = done_ds = None
        idx = 0
        buf = []

        def _flush():
            nonlocal idx, h5_f, action_ds, reward_ds, done_ds
            if not buf:
                return
            b = len(buf)
            n = idx + b

            obs_arrays = {}
            for k in obs_keys_to_save + ['tiago_head_image_preprocessed']:
                frames = [item[0][k] for item in buf if k in item[0]]
                if not frames:
                    continue
                arr = np.array(frames)
                if arr.dtype == object:
                    continue
                obs_arrays[k] = arr

            actions = np.array([item[1] for item in buf])
            rewards = np.array([item[2] for item in buf])
            dones   = np.array([item[3] for item in buf])

            if h5_f is None:
                h5_f = h5py.File(save_path, 'w')
                obs_grp = h5_f.create_group('obs')
                for k, arr in obs_arrays.items():
                    fshape = arr.shape[1:]
                    obs_ds[k] = obs_grp.create_dataset(
                        k, data=arr,
                        maxshape=(None,) + fshape,
                        chunks=(WRITE_BATCH,) + fshape,
                        compression='lzf')
                action_ds = h5_f.create_dataset(
                    'actions', data=actions,
                    maxshape=(None,) + actions.shape[1:], chunks=True)
                reward_ds = h5_f.create_dataset(
                    'rewards', data=rewards, maxshape=(None,), chunks=True)
                done_ds = h5_f.create_dataset(
                    'dones', data=dones, maxshape=(None,), chunks=True)
            else:
                for k, arr in obs_arrays.items():
                    if k not in obs_ds:
                        continue
                    obs_ds[k].resize(n, axis=0)
                    obs_ds[k][idx:n] = arr
                action_ds.resize(n, axis=0); action_ds[idx:n] = actions
                reward_ds.resize(n, axis=0); reward_ds[idx:n] = rewards
                done_ds.resize(n, axis=0);   done_ds[idx:n] = dones

            idx = n
            buf.clear()

        while True:
            item = write_queue.get()
            if item is None:
                _flush()
                if h5_f is not None:
                    h5_f.flush()
                    h5_f.close()
                write_queue.task_done()
                break
            buf.append(item)
            if len(buf) >= WRITE_BATCH:
                _flush()
            write_queue.task_done()

    writer = None
    if save_path is not None:
        writer = threading.Thread(target=_writer_thread, daemon=True)
        writer.start()

    steps_queued = 0
    def append_step(obs_to_save, action_vec, reward, done):
        nonlocal steps_queued
        if writer is None:
            return
        write_queue.put((obs_to_save, action_vec, reward, done))
        steps_queued += 1

    def flush_writer():
        if writer is None:
            return
        write_queue.put(None)
        writer.join()

    if save_vid:
        video = imageio.get_writer(f'{out_dir}/{vid_name}.mp4', fps=10)
        video_full = imageio.get_writer(f'{out_dir}/{vid_name}_full.mp4', fps=10)
    
    is_start = True
    terminated = False

    done = False
    index = 0
    while not rospy.is_shutdown():
        
        action = teleop.get_action(obs)
        buttons = action.extra['buttons']
        
        if (buttons.get('RG', False)):
            index += 1
            #import pdb; pdb.set_trace()
            if terminated:
                print("Should not be here after termination")
                import pdb; pdb.set_trace()
            print("Using policy")
            # Keep raw obs for saving before preprocessing
            raw_obs = {k: obs[k] for k in obs_keys_to_save if k in obs}

            img = _crop_and_resize(obs['tiago_head_image']).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs['tiago_head_image'] = img
            policy_action = model.get_action(obs, batched=False, execute_horizon=execute_horizon).reshape(-1,)

            # Save preprocessed image as (3, 224, 224)
            raw_obs['tiago_head_image_preprocessed'] = img.transpose(2, 0, 1)

            # Save step to HDF5
            append_step(raw_obs, policy_action, 0.0, False)

            if is_start:
                cv2.imwrite(f"test_img_{vid_name}_full.png", raw_obs['tiago_head_image'].astype(np.uint8))
                cv2.imwrite(f"test_img_{vid_name}.png", obs['tiago_head_image'].astype(np.uint8))
                is_start = False

            if SINGLE_HAND:
                action['left'] = policy_action
                action['right'] = np.array([0, 0, 0, 0, 0, 0, 1])
            else:
                right_act = np.concatenate((action[3:9], np.clip([action[15]], 0, 1)))
                left_act = np.concatenate((action[9:15], np.clip([action[16]], 0, 1)))
                action['left'] = left_act
                print('left', left_act)
            
        elif (buttons.get('A', False)):
            done = True
            left_act = np.array([0, 0, 0, 0, 0, 0, 1])
            action['left'] = left_act
            action['right'] = np.array([0, 0, 0, 0, 0, 0, 1])
            #continue
            #policy.start_episode()
        else:
            if not is_start:
                print("Terminating rollout")
                terminated = True

            continue

        print(f"Providing action {action}")    
        n_obs, reward,  _, _, info = env.step(action)
        #done = buttons.get('A', False)

        if done:
            break
        
        obs = copy.deepcopy(n_obs)

        if save_vid:
            full_img = cv2.cvtColor(obs['tiago_head_image'].astype(np.uint8), cv2.COLOR_BGR2RGB)
            video_full.append_data(full_img)
            img = _crop_and_resize(obs['tiago_head_image'])
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            video.append_data(img)

    if save_vid:
        video.close()
        video_full.close()
    flush_writer()
    if save_path is not None:
        print(f'Saved {steps_queued} steps to {save_path}')
    teleop.stop()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    parser.add_argument("--vid_name", default=None, type=str, help="name of the video file (auto-set to rollout_{n} if omitted)")
    parser.add_argument("--out_dir", required=True, default="./", type=str, help="output directory for video")
    parser.add_argument("--save_dir", default=None, type=str, help="directory to save rollout data as HDF5")
    parser.add_argument("--execute_horizon", default=None, type=int, help="number of actions to execute before replanning (default: use all predicted actions)")
    args = parser.parse_args()

    rollout_policy(args.ckpt, args.save_vid, args.vid_name, args.out_dir, args.save_dir, args.execute_horizon)