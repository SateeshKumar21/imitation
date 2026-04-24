"""
Evaluate a trained model at uniformly spaced steps across demos in an HDF5 dataset.

Mirrors the observation / call pattern used in rollout_diffusion_policy.py so the
MSE numbers reflect what the rollout loop actually sees:
  * model.get_action(obs, batched=False, execute_horizon=execute_horizon).reshape(-1,)
  * optional inverse transform of camera-frame policy outputs back into torso frame

Reads the config to determine:
  - data path(s)
  - observation keys + modalities
  - policy class

Finds the best (latest epoch) checkpoint in the experiment directory
and evaluates it against ground truth actions.

Usage:
    python -m imitation.scripts.evaluate_model --config <config_module_path> [options]

Examples:
    # Basic evaluation, primary-frame MSE only
    python -m imitation.scripts.evaluate_model \
        --config imitation.config.model.diffusion_policy_ws5_27th_feb_exp3 \
        --num_eval_points 10 --demos 11 12 13

    # Dual-frame: main h5 (from config) has cam-frame GT; paired h5 has torso-frame GT.
    # Reports MSE before (cam vs cam) and after inverse transform (torso vs torso).
    python -m imitation.scripts.evaluate_model \
        --config imitation.config.model.diffusion_policy_cam_exp \
        --cam_actions \
        --torso_actions_h5 /.../data_1_to_37_notp.h5 \
        --execute_horizon 8
"""
import argparse
import importlib
import os
import re
import sys
from collections import deque, OrderedDict

import h5py
import numpy as np
import torch

ACTION_TRANSFORM_PATH = "/home/ec2-user/action-transformation"
if ACTION_TRANSFORM_PATH not in sys.path:
    sys.path.insert(0, ACTION_TRANSFORM_PATH)

from calibration import T_BASE_TORSO, T_BASE_CAMERA
from transform_il_data_to_camera_frame import invert_transform
from inverse_transform_actions import inverse_transform_actions
from transform_il_data_to_camera_frame import transform_eef_pose


def _load_cam_transforms(calib_npz=None):
    """Return (T_cam_base, T_cam_torso) for cam-frame obs/action transforms."""
    if calib_npz is not None:
        d = np.load(calib_npz)
        T_base_torso = np.asarray(d["T_base_torso"], dtype=np.float64)
        T_base_camera = np.asarray(d["T_base_camera"], dtype=np.float64)
    else:
        T_base_torso = T_BASE_TORSO.copy()
        T_base_camera = T_BASE_CAMERA.copy()
    T_cam_base = invert_transform(T_base_camera)
    T_cam_torso = T_cam_base @ T_base_torso
    return T_cam_base, T_cam_torso


def find_checkpoint(experiment_dir, epoch=None):
    """Find a checkpoint in the experiment directory.
    If epoch is given, look for that specific epoch. Otherwise, pick the latest.
    """
    weights_dir = os.path.join(experiment_dir, "weights")
    if not os.path.isdir(weights_dir):
        return None

    pattern = re.compile(r"weights_ep(\d+)\.pth$")
    candidates = []
    for f in os.listdir(weights_dir):
        m = pattern.match(f)
        if m:
            candidates.append((int(m.group(1)), os.path.join(weights_dir, f)))

    if not candidates:
        return None

    if epoch is not None:
        for ep, path in candidates:
            if ep == epoch:
                return path
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_experiment_dir(config_module_name, base_dir="experiments"):
    """Heuristic: derive experiment directory name from config module name."""
    # e.g. "imitation.config.model.diffusion_policy_ws5_27th_feb_exp3"
    #   -> look for experiments/diffpol_ws5_27th_feb_exp3_*
    short_name = config_module_name.split(".")[-1]

    # Try common naming patterns
    prefixes = [
        short_name,
        short_name.replace("diffusion_policy", "diffpol"),
    ]

    if os.path.isdir(base_dir):
        for entry in sorted(os.listdir(base_dir)):
            entry_lower = entry.lower()
            for prefix in prefixes:
                if prefix.lower() in entry_lower:
                    candidate = os.path.join(base_dir, entry)
                    if find_checkpoint(candidate) is not None:
                        return candidate
    return None


def get_obs_keys_and_modalities(observation_config):
    """Extract obs keys to use and their modalities from the observation config."""
    obs_keys = []
    keys_to_modality = OrderedDict()
    for modality in ["low_dim", "rgb", "depth"]:
        for key in observation_config.obs.get(modality, []):
            obs_keys.append(key)
            keys_to_modality[key] = modality
    return obs_keys, keys_to_modality


def list_demos(hdf5_file):
    """List all demo keys in the HDF5 file, sorted by index."""
    if "data" not in hdf5_file:
        return []
    demos = list(hdf5_file["data"].keys())
    demos.sort(key=lambda x: int(x.split("_")[-1]))
    return demos


def evaluate(config, ckpt_path, demo_indices, num_eval_points,
             execute_horizon=None, cam_actions=False, calib=None,
             torso_actions_h5=None, verbose=True):
    """Run evaluation and return results dict.

    Mirrors the preprocessing / call pattern used in rollout_diffusion_policy.py:
      * `model.get_action(obs, batched=False, execute_horizon=execute_horizon).reshape(-1,)`
      * If `cam_actions`, inverse-transforms each prediction from camera frame to torso frame
        using the calibration in action-transformation/calibration.py (or a --calib .npz).

    Frame conventions:
      * Main data h5 (from config): contains obs + GT actions in the **policy's training frame**.
        The primary MSE is computed against these.
      * `torso_actions_h5` (optional): paired h5 with the same demo keys/lengths but actions in
        torso frame. When combined with `cam_actions`, computes a second "after transformation"
        MSE of `inverse_transform(pred_cam)` vs torso-frame GT.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_class = config.policy_config.policy_class
    model = policy_class.load_weights(ckpt_path)
    model.to(device)

    obs_keys, _ = get_obs_keys_and_modalities(config.observation_config)
    n_obs_steps = getattr(model, "n_obs_steps", 1)

    data_paths = config.data_config.data
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    T_cam_base = T_cam_torso = None
    if cam_actions:
        T_cam_base, T_cam_torso = _load_cam_transforms(calib)
        if verbose:
            print(f"[cam_actions] inverse-transforming camera-frame policy actions to torso frame "
                  f"(calib={'builtin' if calib is None else calib})")

    dual_frame = cam_actions and torso_actions_h5 is not None
    torso_hdf5 = h5py.File(torso_actions_h5, "r") if torso_actions_h5 is not None else None
    if dual_frame and verbose:
        print(f"[dual_frame] also computing torso-frame MSE against {torso_actions_h5}")

    results = {}

    try:
        for data_path in data_paths:
            hdf5 = h5py.File(data_path, "r")
            available_demos = list_demos(hdf5)
            available_indices = [int(d.split("_")[-1]) for d in available_demos]

            # Decide once per h5 whether obs need on-the-fly base->camera transformation.
            h5_frame = hdf5.attrs.get("frame")
            if isinstance(h5_frame, bytes):
                h5_frame = h5_frame.decode()
            transform_obs = cam_actions and h5_frame != "camera"
            if verbose:
                if transform_obs:
                    print(f"[cam_obs] transforming obs/left,obs/right base->camera for {data_path}")
                elif cam_actions:
                    print(f"[cam_obs] {data_path} already in camera frame (frame attr); obs passthrough")

            if demo_indices is not None:
                run_indices = [i for i in demo_indices if i in available_indices]
            else:
                run_indices = available_indices

            for demo_idx in run_indices:
                demo_key = f"data/demo_{demo_idx}"
                demo = hdf5[demo_key]
                demo_len = demo["actions"].shape[0]

                # Fetch paired torso-frame GT for this demo once, up front.
                torso_demo_actions = None
                if dual_frame:
                    if demo_key not in torso_hdf5:
                        if verbose:
                            print(f"  [warn] {demo_key} missing from torso h5; skipping torso metric.")
                    else:
                        torso_demo_actions = torso_hdf5[demo_key]["actions"]
                        if torso_demo_actions.shape[0] != demo_len:
                            raise ValueError(f"{demo_key}: length mismatch main={demo_len} "
                                             f"torso={torso_demo_actions.shape[0]}")

                eval_indices = np.linspace(0, demo_len - 1, num_eval_points, dtype=int)

                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Demo {demo_idx} (length={demo_len}), eval at: {eval_indices.tolist()}")
                    print(f"{'='*70}")

                predicted = []          # policy-frame (cam if cam_actions, else torso)
                gt_actions = []         # matches frame of `predicted`
                predicted_torso = []    # only populated when dual_frame
                gt_torso = []

                for target_idx in eval_indices:
                    # Reset model queues for each eval point (independent evaluation).
                    model.obs_queue = deque(maxlen=n_obs_steps)
                    model.action_queue = deque()

                    # Feed a window of observations ending at target_idx
                    window_start = max(0, target_idx - n_obs_steps + 1)
                    for t in range(window_start, target_idx + 1):
                        obs = {}
                        for key in obs_keys:
                            val = np.array(demo[f"obs/{key}"][t])
                            if transform_obs and key in ("left", "right") and val.shape == (8,):
                                val = transform_eef_pose(val.reshape(1, -1), T_cam_base)[0].astype(val.dtype)
                            obs[key] = val
                        with torch.no_grad():
                            action = model.get_action(obs, batched=False,
                                                      execute_horizon=execute_horizon)

                    pred_action = np.asarray(action).reshape(-1,)
                    gt_action = np.array(demo["actions"][target_idx])

                    predicted.append(pred_action)
                    gt_actions.append(gt_action)

                    # Dual-frame: inverse-transform the camera-frame prediction into torso frame
                    # and fetch the paired torso-frame GT.
                    if dual_frame and torso_demo_actions is not None:
                        pred_torso_vec = inverse_transform_actions(
                            pred_action[None, :], T_cam_torso)[0].astype(pred_action.dtype)
                        gt_torso_vec = np.array(torso_demo_actions[target_idx])
                        predicted_torso.append(pred_torso_vec)
                        gt_torso.append(gt_torso_vec)

                    if verbose:
                        err = np.abs(pred_action - gt_action)
                        fmt = lambda a: np.array2string(a, precision=4, suppress_small=True, max_line_width=200)
                        frame_tag = "cam" if cam_actions else "torso"
                        print(f"  idx={target_idx:>4d}  pred[{frame_tag}]={fmt(pred_action)}")
                        print(f"          GT  [{frame_tag}]={fmt(gt_action)}")
                        print(f"          |err|       ={fmt(err)}")
                        if dual_frame and predicted_torso:
                            err_t = np.abs(predicted_torso[-1] - gt_torso[-1])
                            print(f"          pred[torso]={fmt(predicted_torso[-1])}")
                            print(f"          GT  [torso]={fmt(gt_torso[-1])}")
                            print(f"          |err|      ={fmt(err_t)}")

                predicted = np.array(predicted)
                gt_actions = np.array(gt_actions)
                mse = float(np.mean((predicted - gt_actions) ** 2))
                mae = float(np.mean(np.abs(predicted - gt_actions)))
                per_dim_mse = np.mean((predicted - gt_actions) ** 2, axis=0)

                demo_result = {
                    "demo_len": demo_len,
                    "n_eval_points": len(eval_indices),
                    "mse": mse,
                    "mae": mae,
                    "per_dim_mse": per_dim_mse,
                    "frame": "cam" if cam_actions else "torso",
                }

                if dual_frame and predicted_torso:
                    predicted_torso = np.array(predicted_torso)
                    gt_torso = np.array(gt_torso)
                    demo_result["mse_torso"] = float(np.mean((predicted_torso - gt_torso) ** 2))
                    demo_result["mae_torso"] = float(np.mean(np.abs(predicted_torso - gt_torso)))
                    demo_result["per_dim_mse_torso"] = np.mean((predicted_torso - gt_torso) ** 2, axis=0)

                results[demo_idx] = demo_result

                if verbose:
                    line = f"  => [cam-frame ] MSE={mse:.6f}  MAE={mae:.6f}" if cam_actions \
                           else f"  => MSE={mse:.6f}  MAE={mae:.6f}"
                    print(line)
                    if "mse_torso" in demo_result:
                        print(f"  => [torso-frame] MSE={demo_result['mse_torso']:.6f}  "
                              f"MAE={demo_result['mae_torso']:.6f}")

            hdf5.close()
    finally:
        if torso_hdf5 is not None:
            torso_hdf5.close()

    return results


def print_summary(results):
    """Print a formatted summary table. If dual-frame metrics are present,
    also print the torso-frame (after inverse transform) summary."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    any_dual = any("mse_torso" in r for r in results.values())
    primary_frame = next((r["frame"] for r in results.values() if "frame" in r), "policy")

    header = f"{'Demo':>4s} {'len':>4s} | "
    header += f"{primary_frame+'_MSE':>14s} {primary_frame+'_MAE':>14s}"
    if any_dual:
        header += f" | {'torso_MSE':>14s} {'torso_MAE':>14s}"
    print(header)
    print("-" * len(header))

    mses, maes, all_per_dim = [], [], []
    mses_t, maes_t, all_per_dim_t = [], [], []
    for demo_idx in sorted(results.keys()):
        r = results[demo_idx]
        row = f"{demo_idx:>4d} {r['demo_len']:>4d} | {r['mse']:>14.6f} {r['mae']:>14.6f}"
        if any_dual and "mse_torso" in r:
            row += f" | {r['mse_torso']:>14.6f} {r['mae_torso']:>14.6f}"
        elif any_dual:
            row += f" | {'-':>14s} {'-':>14s}"
        print(row)

        mses.append(r["mse"])
        maes.append(r["mae"])
        all_per_dim.append(r["per_dim_mse"])
        if "mse_torso" in r:
            mses_t.append(r["mse_torso"])
            maes_t.append(r["mae_torso"])
            all_per_dim_t.append(r["per_dim_mse_torso"])

    if mses:
        print()
        print(f"  [{primary_frame}] avg MSE: {np.mean(mses):.6f} (+/- {np.std(mses):.6f})")
        print(f"  [{primary_frame}] avg MAE: {np.mean(maes):.6f} (+/- {np.std(maes):.6f})")
        print(f"  [{primary_frame}] per-dim avg MSE: "
              f"{np.array2string(np.mean(all_per_dim, axis=0), precision=6)}")
    if mses_t:
        print()
        print(f"  [torso ] avg MSE: {np.mean(mses_t):.6f} (+/- {np.std(mses_t):.6f})")
        print(f"  [torso ] avg MAE: {np.mean(maes_t):.6f} (+/- {np.std(maes_t):.6f})")
        print(f"  [torso ] per-dim avg MSE: "
              f"{np.array2string(np.mean(all_per_dim_t, axis=0), precision=6)}")
    if mses:
        print(f"\n  Demos evaluated: {len(mses)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model against GT actions.")
    parser.add_argument("--config", type=str, required=True,
                        help="Config module path, e.g. imitation.config.model.diffusion_policy_ws5_27th_feb_exp3")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. If not given, auto-discovers from experiment dir.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch to evaluate. Used with auto-discovery.")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Experiment directory to search for checkpoints.")
    parser.add_argument("--demos", type=int, nargs="+", default=None,
                        help="Demo indices to evaluate. Default: all demos in the dataset.")
    parser.add_argument("--num_eval_points", type=int, default=10,
                        help="Number of uniformly spaced eval points per demo.")
    parser.add_argument("--execute_horizon", type=int, default=None,
                        help="Passed to model.get_action() (mirrors rollout_diffusion_policy.py).")
    parser.add_argument("--cam_actions", action="store_true",
                        help="Policy emits camera-frame actions; inverse-transform to torso frame.")
    parser.add_argument("--calib", type=str, default=None,
                        help=".npz with T_base_torso, T_base_camera (overrides built-in calibration).")
    parser.add_argument("--torso_actions_h5", type=str, default=None,
                        help="Paired h5 with torso-frame GT actions (same demos/lengths as the main "
                             "data h5). Use with --cam_actions to compute both before-transform "
                             "(cam vs cam) and after-transform (torso vs torso) MSE.")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Override config.data_config.data with these h5 paths (eval on a "
                             "different dataset than the one the model was trained on).")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, not per-step details.")
    args = parser.parse_args()

    if args.torso_actions_h5 is not None and not args.cam_actions:
        print("WARNING: --torso_actions_h5 has no effect without --cam_actions; ignoring.")
        args.torso_actions_h5 = None

    # Load config
    config_module = importlib.import_module(args.config)
    config = config_module.config

    if args.data is not None:
        print(f"[override] config.data_config.data -> {args.data}")
        config.data_config.data = args.data

    # Find checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        exp_dir = args.experiment_dir
        if exp_dir is None:
            exp_dir = find_experiment_dir(args.config)
        if exp_dir is None:
            print(f"ERROR: Could not auto-discover experiment dir for '{args.config}'.")
            print("       Pass --ckpt or --experiment_dir explicitly.")
            sys.exit(1)
        ckpt_path = find_checkpoint(exp_dir, epoch=args.epoch)
        if ckpt_path is None:
            print(f"ERROR: No checkpoint found in '{exp_dir}'.")
            sys.exit(1)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Eval points per demo: {args.num_eval_points}")
    if args.demos:
        print(f"Demos: {args.demos}")
    else:
        print(f"Demos: all")

    results = evaluate(
        config=config,
        ckpt_path=ckpt_path,
        demo_indices=args.demos,
        num_eval_points=args.num_eval_points,
        execute_horizon=args.execute_horizon,
        cam_actions=args.cam_actions,
        calib=args.calib,
        torso_actions_h5=args.torso_actions_h5,
        verbose=not args.quiet,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
