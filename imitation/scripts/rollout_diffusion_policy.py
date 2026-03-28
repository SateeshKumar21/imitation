import os
import time
import cv2
import copy
import rospy
import numpy as np
import imageio
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
from gymnasium.wrappers import FrameStackObservation
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


SINGLE_HAND=True
def rollout_policy(model_ckpt, save_vid=False, vid_name=None, out_dir="./"):

    # load policy
    os.makedirs(out_dir, exist_ok=True)
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

    if save_vid:
        video = imageio.get_writer(f'{out_dir}/{vid_name}.mp4', fps=10)
    
    is_start = True
    terminated = False
    done = False
    while not rospy.is_shutdown():
        
        action = teleop.get_action(obs)
        buttons = action.extra['buttons']
        
        if (buttons.get('RG', False)):
            #import pdb; pdb.set_trace()
            if terminated:
                print("Should not be here after termination")
                import pdb; pdb.set_trace()
            print("Using policy")
            obs['tiago_head_image'] = _crop_and_resize(obs['tiago_head_image']).astype(np.float16)
            policy_action = model.get_action(obs, batched=False).reshape(-1,)

       
            if is_start:
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
            img = _crop_and_resize(obs['tiago_head_image'])
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            video.append_data(img)

    if save_vid:
        video.close()
    teleop.stop()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    parser.add_argument("--vid_name", required=True, type=str, help="name of the video file")
    parser.add_argument("--out_dir", required=True, default="./", type=str, help="output directory for video")
    args = parser.parse_args()

    rollout_policy(args.ckpt, args.save_vid, args.vid_name, args.out_dir)