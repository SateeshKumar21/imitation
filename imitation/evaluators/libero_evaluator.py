import os
import numpy as np
import gym
from tqdm import tqdm
import robosuite.utils.transform_utils as T
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
'''
Task ID to instruction,
0 put both the alphabet soup and the tomato sauce in the basket
1 put both the cream cheese box and the butter in the basket
2 turn on the stove and put the moka pot on it
3 put the black bowl in the bottom drawer of the cabinet and close it
4 put the white mug on the left plate and put the yellow and white mug on the right plate
5 pick up the book and place it in the back compartment of the caddy
6 put the white mug on the plate and put the chocolate pudding to the right of the plate
7 put both the alphabet soup and the cream cheese box in the basket
8 put both moka pots on the stove
9 put the yellow and white mug in the microwave and close it
'''


def write_video(video_path, frames, fps=30):
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

class EvalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def process_obs(self, obs):
        processed_obs = {
            'ee_pos': np.array(obs['robot0_eef_pos']).astype(np.float32), 
            'ee_ori': np.array(T.quat2axisangle(obs['robot0_eef_quat'])).astype(np.float32),
            'gripper_states': np.array(obs['robot0_gripper_qpos']).astype(np.float32),
            # 'robot_states': np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]], axis=-1).astype(np.float32),
            'agentview_rgb': np.array(obs['agentview_image']).astype(float),
            'eye_in_hand_rgb': np.array(obs['robot0_eye_in_hand_image']).astype(float)
        }

        return processed_obs
    
    def step(self, action):
        n_obs, reward, done, info = self.env.step(action)
        n_obs = self.process_obs(n_obs)
        
        return n_obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        obs = self.process_obs(obs)
        return obs

class Libero10EnvGenerator:
    def __init__(self, num_env=5, wrapper_kwargs={}):
        self.task_suite = benchmark.get_benchmark("libero_10")()
        self.num_env = num_env
        self.wrapper_kwargs = wrapper_kwargs
        
        self.load_all_tasks()


    def load_all_tasks(self):
        self.env_args = []
        self.task_instructions = []
        self.init_states = []
        for task_id in range(10):
            task = self.task_suite.get_task(task_id)
            task_description = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self.env_args.append({
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            })
            self.task_instructions.append(task_description)
            self.init_states.append(self.task_suite.get_task_init_states(task_id))
    
    def get_env(self, task_id):
        env = SubprocVectorEnv(
            [lambda: EvalWrapper(OffScreenRenderEnv(**self.env_args[task_id])) for _ in range(self.num_env)]
        )
        
        return env
    
    def set_init_state(self, env, task_id, init_set):
        env.reset()
        env.seed(0)

        init_states = self.init_states[task_id]
        indices = np.arange(init_set*self.num_env, (init_set+1)*self.num_env) % init_states.shape[0]
        init_states_ = init_states[indices]
        env.set_init_state(init_states_)

        dummy_action = np.zeros((self.num_env, 7))
        dummy_action[..., -1] = -1
        for _ in range(20):  # simulate the physics without any actions
            init_obs, _, _, _ = env.step(dummy_action)
        
        return init_obs


def stack_obs(obs):
    obs = {k: np.stack([o[k] for o in obs], axis=0) for k in obs[0].keys()}
    return obs

class LiberoEvaluator:

    def __init__(self, eval_config, *args, **kwargs):
        self.eval_config = eval_config

        self.env_gen = Libero10EnvGenerator(num_env=self.eval_config.num_envs)
        self.env = self.env_gen.get_env(self.eval_config.task_id)

    def evaluate(self, policy):
        total_evals = self.eval_config.total_evals
        num_envs = self.eval_config.num_envs
        language_instruction = self.env_gen.task_instructions[self.eval_config.task_id]

        task_success = 0

        for init_set in range(total_evals//num_envs):
            init_obs = self.env_gen.set_init_state(self.env, self.eval_config.task_id, init_set)
            success_counts, video = self.rollout_policy(policy, init_obs)
            task_success += success_counts

            if self.eval_config.save_vid:
                video_path = f"{self.eval_config.out_dir}/{language_instruction}_{init_set}.mp4"
                write_video(video_path, np.flip(video, axis=1).astype(np.uint8))

        return {"eval_success": task_success / total_evals}
    
    def rollout_policy(self, policy, obs):
    
        num_envs = self.eval_config.num_envs
        done_mask = np.zeros(num_envs, dtype=bool)
        images = []

        obs = stack_obs(obs)
        images.append(obs['agentview_rgb'][0])
        for i in tqdm(range(self.eval_config.num_steps), desc="Rollout", total=self.eval_config.num_steps):

            action = np.array(policy.get_action(obs, batched=True), dtype=np.float64)

            obs, _, _, _ = self.env.step(action)
            dones = np.array(self.env.check_success())
            done_mask = np.logical_or(done_mask, dones)
            if np.all(done_mask):
                break

            obs = stack_obs(obs)
            images.append(obs['agentview_rgb'][0])
            if done_mask[0]:
                # highlight green if success
                images[-1] = (images[-1] + np.array([0, 255, 0])) / 2
            if np.all(done_mask):
                return np.sum(done_mask), images

        return np.sum(done_mask), images

if __name__ == "__main__":
    from imitation.utils.general_utils import AttrDict
    eval_config = AttrDict(
        task_id=5,  # Change this to the desired task ID
        num_envs=1,
        total_evals=5,
        num_steps=100,
        save_vid=True,
        out_dir='/home/shivin/compi/experiments/videos',
    )

    # Load the policy
    from imitation.algo.base_algo import BaseAlgo

    base_path = '/home/shivin/compi/experiments/compi_book-caddy/weights'
    # base_path = '/home/shivin/compi/experiments/flow_book-caddy_unet/weights'
    # base_path = '/home/shivin/compi/experiments/diffusion_book-caddy_unet3/weights'

    weight_list = os.listdir(base_path)
    weight_list = sorted(weight_list, key=lambda x: int(x.split('_')[1][2:].split('.')[0]), reverse=True)[:1]

    evaluator = LiberoEvaluator(eval_config)
    for weight in weight_list:
        print(f"Loading weights: {weight}")
        weight_path = os.path.join(base_path, weight)
        policy = BaseAlgo.load_weights(weight_path)
        policy.to('cuda')

        results = evaluator.evaluate(policy)
        print(f"Task Success Rate: {results['task_success']:.2f}")